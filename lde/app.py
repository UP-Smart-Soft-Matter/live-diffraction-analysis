"""Live diffraction-efficiency GUI."""
from __future__ import annotations

import csv
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import tifffile
from PySide6 import QtCore, QtGui, QtWidgets

from .analysis import AnalysisParams, AnalysisResult, analyze
from .cameras import (Camera, CameraError, SimulatedCamera, ThorlabsCamera,
                      list_thorlabs_cameras, shutdown_sdk)

pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)

SIM_LABEL = "Simulated camera"
HISTORY_LEN = 600
ORDER_COLORS = ["#ffffff", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#e377c2", "#bcbd22", "#17becf", "#1f77b4"]  # index = order mod 9


class Acquisition(threading.Thread):
    """Pulls frames from the camera and keeps only the newest one."""

    def __init__(self, cam: Camera):
        super().__init__(daemon=True)
        self.cam = cam
        self._lock = threading.Lock()
        self._frame = None
        self._count = 0
        self._stop_evt = threading.Event()
        self.error: str | None = None

    def run(self):
        try:
            self.cam.start()
            while not self._stop_evt.is_set():
                frame = self.cam.read()
                if frame is not None:
                    with self._lock:
                        self._frame = frame
                        self._count += 1
        except Exception as e:  # camera unplugged etc.
            self.error = str(e)
        finally:
            try:
                self.cam.stop()
            except Exception:
                pass

    def latest(self):
        with self._lock:
            return self._frame, self._count

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=3)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, sim_image: np.ndarray | None = None):
        super().__init__()
        self.setWindowTitle("Live Diffraction Efficiency")
        self.resize(1500, 900)

        self.sim_image = sim_image
        self.cam: Camera | None = None
        self.acq: Acquisition | None = None
        self.params = AnalysisParams()
        self.result = AnalysisResult()
        self.frame: np.ndarray | None = None      # processed frame that is shown/analysed
        self._undarked: np.ndarray | None = None  # averaged frame before dark subtraction
        self._dark: np.ndarray | None = None
        self._avg = deque(maxlen=1)
        self._last_count = 0
        self._locked_peaks: np.ndarray | None = None  # full-frame (row, col)
        self._history = deque(maxlen=HISTORY_LEN)
        self._t0 = time.perf_counter()
        self._fps_times = deque(maxlen=30)
        self._log_file = None
        self._log_writer = None
        self._bar_labels: list[pg.TextItem] = []
        self._hist_curves: dict[int, pg.PlotDataItem] = {}

        self._build_plots()
        self._build_controls()
        self.refresh_cameras(silent=True)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    # ------------------------------------------------------------------ UI
    def _build_plots(self):
        glw = pg.GraphicsLayoutWidget()
        self.setCentralWidget(glw)

        self.img_plot = glw.addPlot(row=0, col=0, rowspan=3, title="Camera")
        self.img_plot.setAspectLocked(True)
        self.img_plot.invertY(True)
        self.img_item = pg.ImageItem(autoDownsample=True)
        self.img_item.setColorMap(pg.colormap.get("viridis"))
        self.img_plot.addItem(self.img_item)

        self.crop_roi = pg.RectROI([0, 0], [100, 100], pen=pg.mkPen("w", width=1, style=QtCore.Qt.DashLine))
        self.crop_roi.addScaleHandle([0, 0], [1, 1])
        self.crop_roi.setZValue(10)
        self.img_plot.addItem(self.crop_roi)

        self.roi_boxes = pg.PlotCurveItem(pen=pg.mkPen("r", width=1), connect="finite")
        self.line_item = pg.PlotCurveItem(pen=pg.mkPen((170, 220, 255), width=1))
        self.peak_item = pg.ScatterPlotItem(size=5, brush="r", pen=None)
        for it in (self.roi_boxes, self.line_item, self.peak_item):
            it.setZValue(20)
            self.img_plot.addItem(it)
        self.order_texts: list[pg.TextItem] = []

        self.bar_plot = glw.addPlot(row=0, col=1, title="Diffraction efficiency")
        self.bar_plot.setLabel("bottom", "Diffraction order")
        self.bar_plot.setLabel("left", "Efficiency", units="%")
        self.bar_plot.setMouseEnabled(x=False, y=False)
        self.bar_item = pg.BarGraphItem(x=[], height=[], width=0.6, brush=(80, 140, 220))
        self.bar_plot.addItem(self.bar_item)

        self.prof_plot = glw.addPlot(row=1, col=1, title="Profile along the orders")
        self.prof_plot.setLabel("left", "Counts")
        self.prof_curve = self.prof_plot.plot(pen="y")

        self.hist_plot = glw.addPlot(row=2, col=1, title="Efficiency over time")
        self.hist_plot.setLabel("bottom", "Time", units="s")
        self.hist_plot.setLabel("left", "Efficiency", units="%")
        self.hist_plot.addLegend(offset=(-5, 5), colCount=5)

        glw.ci.layout.setColumnStretchFactor(0, 3)
        glw.ci.layout.setColumnStretchFactor(1, 2)

        self.status_cam = QtWidgets.QLabel("Not connected")
        self.status_sat = QtWidgets.QLabel("")
        self.statusBar().addWidget(self.status_cam, 1)
        self.statusBar().addPermanentWidget(self.status_sat)

    def _build_controls(self):
        dock = QtWidgets.QDockWidget("Controls", self)
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable)
        panel = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(panel)

        # --- camera
        box = QtWidgets.QGroupBox("Camera")
        form = QtWidgets.QFormLayout(box)
        self.cam_combo = QtWidgets.QComboBox()
        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.refresh_btn.clicked.connect(lambda: self.refresh_cameras(silent=False))
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.cam_combo, 1)
        row.addWidget(self.refresh_btn)
        form.addRow(row)
        self.connect_btn = QtWidgets.QPushButton("Connect")
        self.connect_btn.clicked.connect(self.toggle_connection)
        form.addRow(self.connect_btn)
        self.exp_spin = QtWidgets.QDoubleSpinBox(decimals=3, suffix=" ms")
        self.exp_spin.setRange(0.001, 10000)
        self.exp_spin.setKeyboardTracking(False)
        self.exp_spin.valueChanged.connect(self._set_exposure)
        form.addRow("Exposure", self.exp_spin)
        self.gain_spin = QtWidgets.QSpinBox()
        self.gain_spin.setKeyboardTracking(False)
        self.gain_spin.valueChanged.connect(self._set_gain)
        form.addRow("Gain", self.gain_spin)
        self.avg_spin = QtWidgets.QSpinBox()
        self.avg_spin.setRange(1, 100)
        self.avg_spin.valueChanged.connect(self._set_averaging)
        form.addRow("Average frames", self.avg_spin)
        self.autolevel_chk = QtWidgets.QCheckBox("Auto contrast")
        self.autolevel_chk.setChecked(True)
        form.addRow(self.autolevel_chk)
        self.dark_btn = QtWidgets.QPushButton("Capture dark frame")
        self.dark_btn.clicked.connect(self.capture_dark)
        self.dark_clear_btn = QtWidgets.QPushButton("Clear")
        self.dark_clear_btn.clicked.connect(self.clear_dark)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.dark_btn, 1)
        row.addWidget(self.dark_clear_btn)
        form.addRow(row)
        outer.addWidget(box)

        # --- analysis
        box = QtWidgets.QGroupBox("Analysis")
        form = QtWidgets.QFormLayout(box)
        self.roi_spin = QtWidgets.QSpinBox()
        self.roi_spin.setRange(1, 500)
        self.roi_spin.setValue(self.params.roi_half)
        self.roi_spin.setSuffix(" px")
        form.addRow("ROI half-size", self.roi_spin)
        self.dist_spin = QtWidgets.QSpinBox()
        self.dist_spin.setRange(1, 2000)
        self.dist_spin.setValue(self.params.min_distance)
        self.dist_spin.setSuffix(" px")
        form.addRow("Min. order spacing", self.dist_spin)
        self.thr_spin = QtWidgets.QDoubleSpinBox(decimals=1, suffix=" %")
        self.thr_spin.setRange(0.0, 100.0)
        self.thr_spin.setSingleStep(0.5)
        self.thr_spin.setValue(self.params.threshold_rel * 100)
        form.addRow("Peak threshold", self.thr_spin)
        self.smooth_spin = QtWidgets.QDoubleSpinBox(decimals=1, suffix=" px")
        self.smooth_spin.setRange(0.0, 50.0)
        self.smooth_spin.setSingleStep(0.5)
        self.smooth_spin.setValue(self.params.smooth_sigma)
        form.addRow("Peak smoothing", self.smooth_spin)
        self.max_spin = QtWidgets.QSpinBox()
        self.max_spin.setRange(0, 99)
        self.max_spin.setSpecialValueText("all")
        form.addRow("Max. orders", self.max_spin)
        self.zero_combo = QtWidgets.QComboBox()
        self.zero_combo.addItem("Middle peak", "center")
        self.zero_combo.addItem("Brightest peak", "brightest")
        form.addRow("Zero order", self.zero_combo)
        self.bg_chk = QtWidgets.QCheckBox("Subtract background (median)")
        form.addRow(self.bg_chk)
        self.lock_chk = QtWidgets.QCheckBox("Lock order positions")
        self.lock_chk.toggled.connect(self._toggle_lock)
        form.addRow(self.lock_chk)
        self.reset_crop_btn = QtWidgets.QPushButton("Reset search region")
        self.reset_crop_btn.clicked.connect(self.reset_crop)
        form.addRow(self.reset_crop_btn)
        outer.addWidget(box)

        # --- output
        box = QtWidgets.QGroupBox("Output")
        v = QtWidgets.QVBoxLayout(box)
        self.snap_btn = QtWidgets.QPushButton("Save snapshot (TIFF + CSV)…")
        self.snap_btn.clicked.connect(self.save_snapshot)
        v.addWidget(self.snap_btn)
        self.log_btn = QtWidgets.QPushButton("Record to CSV…")
        self.log_btn.setCheckable(True)
        self.log_btn.clicked.connect(self.toggle_logging)
        v.addWidget(self.log_btn)
        self.clear_hist_btn = QtWidgets.QPushButton("Clear time trace")
        self.clear_hist_btn.clicked.connect(self.clear_history)
        v.addWidget(self.clear_hist_btn)
        outer.addWidget(box)

        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Order", "Intensity", "Efficiency"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        outer.addWidget(self.table, 1)

        dock.setWidget(panel)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)
        self._set_connected_ui(False)

    def _set_connected_ui(self, connected: bool):
        self.connect_btn.setText("Disconnect" if connected else "Connect")
        self.cam_combo.setEnabled(not connected)
        self.refresh_btn.setEnabled(not connected)
        for w in (self.exp_spin, self.gain_spin, self.dark_btn, self.snap_btn, self.log_btn):
            w.setEnabled(connected)
        if connected and self.cam.gain_range[1] <= 0:
            self.gain_spin.setEnabled(False)

    # -------------------------------------------------------------- camera
    def refresh_cameras(self, silent: bool):
        self.cam_combo.clear()
        msg = ""
        try:
            for serial in list_thorlabs_cameras():
                self.cam_combo.addItem(f"Thorlabs {serial}", serial)
            if self.cam_combo.count() == 0:
                msg = "No Thorlabs camera found."
        except CameraError as e:
            msg = str(e)
        self.cam_combo.addItem(SIM_LABEL, None)
        if msg:
            self.status_cam.setText(msg.splitlines()[0])
            if not silent:
                QtWidgets.QMessageBox.information(self, "Thorlabs cameras", msg)

    def toggle_connection(self):
        if self.cam is None:
            self.connect_camera()
        else:
            self.disconnect_camera()

    def connect_camera(self):
        serial = self.cam_combo.currentData()
        try:
            cam = SimulatedCamera(self.sim_image) if serial is None else ThorlabsCamera(serial)
        except CameraError as e:
            QtWidgets.QMessageBox.critical(self, "Camera", str(e))
            return
        self.cam = cam
        lo, hi = cam.exposure_range_ms
        for spin, rng, val in ((self.exp_spin, (lo, hi), cam.get_exposure_ms()),
                               (self.gain_spin, cam.gain_range, cam.get_gain())):
            spin.blockSignals(True)
            spin.setRange(*rng)
            spin.setValue(val)
            spin.blockSignals(False)
        self._avg.clear()
        self._dark = None
        self._last_count = 0
        self._first_frame = True
        self.lock_chk.setChecked(False)
        self.clear_history()
        self.acq = Acquisition(cam)
        self.acq.start()
        self._set_connected_ui(True)
        self.status_cam.setText(f"{cam.name} — waiting for frames…")

    def disconnect_camera(self):
        if self.log_btn.isChecked():
            self.log_btn.setChecked(False)
            self.toggle_logging()
        if self.acq is not None:
            self.acq.stop()
            self.acq = None
        if self.cam is not None:
            try:
                self.cam.close()
            except Exception:
                pass
            self.cam = None
        self._set_connected_ui(False)
        self.status_cam.setText("Not connected")
        self.status_sat.setText("")

    def _set_exposure(self, value):
        if self.cam is not None:
            try:
                self.cam.set_exposure_ms(value)
            except Exception as e:
                self.statusBar().showMessage(f"Could not set exposure: {e}", 5000)

    def _set_gain(self, value):
        if self.cam is not None:
            try:
                self.cam.set_gain(value)
            except Exception as e:
                self.statusBar().showMessage(f"Could not set gain: {e}", 5000)

    def _set_averaging(self, n):
        self._avg = deque(self._avg, maxlen=n)

    def capture_dark(self):
        if self._undarked is not None:
            self._dark = self._undarked.copy()
            self.statusBar().showMessage("Dark frame captured — it is subtracted from every frame.", 5000)

    def clear_dark(self):
        self._dark = None

    # ------------------------------------------------------------ analysis
    def _read_params(self):
        p = self.params
        p.roi_half = self.roi_spin.value()
        p.min_distance = self.dist_spin.value()
        p.threshold_rel = self.thr_spin.value() / 100.0
        p.smooth_sigma = self.smooth_spin.value()
        p.max_orders = self.max_spin.value()
        p.zero_order = self.zero_combo.currentData()
        p.subtract_background = self.bg_chk.isChecked()

    def _crop_bounds(self, shape):
        h, w = shape
        pos, size = self.crop_roi.pos(), self.crop_roi.size()
        x0 = int(np.clip(round(pos.x()), 0, w - 1))
        y0 = int(np.clip(round(pos.y()), 0, h - 1))
        x1 = int(np.clip(round(pos.x() + size.x()), x0 + 1, w))
        y1 = int(np.clip(round(pos.y() + size.y()), y0 + 1, h))
        return x0, y0, x1, y1

    def reset_crop(self):
        if self.frame is not None:
            h, w = self.frame.shape
            self.crop_roi.setPos([0, 0])
            self.crop_roi.setSize([w, h])

    def _toggle_lock(self, on: bool):
        if on and self.result.ok and self.frame is not None:
            x0, y0, _, _ = self._crop_bounds(self.frame.shape)
            self._locked_peaks = self.result.peaks + np.array([y0, x0])
        else:
            self._locked_peaks = None
            if on:  # nothing to lock
                self.lock_chk.setChecked(False)

    def update_frame(self):
        if self.acq is None:
            return
        if self.acq.error:
            err = self.acq.error
            self.disconnect_camera()
            QtWidgets.QMessageBox.critical(self, "Camera", f"Acquisition stopped:\n{err}")
            return
        raw, count = self.acq.latest()
        if raw is None or count == self._last_count:
            return
        self._last_count = count
        now = time.perf_counter()
        self._fps_times.append(now)

        self._avg.append(raw.astype(np.float32))
        if len(self._avg) > 1 and self._avg[0].shape != raw.shape:
            self._avg.clear()
            self._avg.append(raw.astype(np.float32))
        frame = self._avg[0] if len(self._avg) == 1 else np.mean(self._avg, axis=0)
        self._undarked = frame
        if self._dark is not None and self._dark.shape == frame.shape:
            frame = frame - self._dark
        self.frame = frame

        if self._first_frame:
            self._first_frame = False
            h, w = frame.shape
            self.crop_roi.maxBounds = QtCore.QRectF(0, 0, w, h)
            self.reset_crop()
            self.img_plot.setRange(xRange=(0, w), yRange=(0, h), padding=0)

        levels = (float(frame.min()), max(float(frame.max()), float(frame.min()) + 1)) \
            if self.autolevel_chk.isChecked() else (0, self.cam.max_value)
        self.img_item.setImage(frame, autoLevels=False, levels=levels)

        self._read_params()
        if self._locked_peaks is not None:
            x0 = y0 = 0
            res = analyze(frame, self.params, peaks=self._locked_peaks)
        else:
            x0, y0, x1, y1 = self._crop_bounds(frame.shape)
            res = analyze(frame[y0:y1, x0:x1], self.params)
        self.result = res

        # saturation is judged on the raw camera frame, inside the ROIs
        saturated = False
        r = self.params.roi_half
        for y, x in res.peaks:
            yy, xx = y + y0, x + x0
            if raw[max(0, yy - r):yy + r, max(0, xx - r):xx + r].max(initial=0) >= self.cam.max_value:
                saturated = True
                break
        if saturated:
            self.status_sat.setText("SATURATED — reduce exposure, efficiencies are invalid")
            self.status_sat.setStyleSheet("color: white; background: #c0392b; padding: 2px 8px; font-weight: bold;")
        else:
            self.status_sat.setText(f"max {int(raw.max())} / {self.cam.max_value}")
            self.status_sat.setStyleSheet("")

        self._draw_overlay(res, x0, y0)
        self._draw_results(res, now - self._t0)
        self._write_log(res, x0, y0)

        fps = 0.0
        if len(self._fps_times) > 1:
            fps = (len(self._fps_times) - 1) / (self._fps_times[-1] - self._fps_times[0])
        self.status_cam.setText(
            f"{self.cam.name} — {frame.shape[1]}×{frame.shape[0]} px, {fps:.1f} fps, "
            f"{len(res.peaks)} orders" + (", dark-subtracted" if self._dark is not None else ""))

    def _draw_overlay(self, res: AnalysisResult, x0: int, y0: int):
        n = len(res.peaks)
        while len(self.order_texts) > n:
            self.img_plot.removeItem(self.order_texts.pop())
        while len(self.order_texts) < n:
            t = pg.TextItem(color="w", anchor=(0.5, 1.0))
            t.setZValue(21)
            self.img_plot.addItem(t)
            self.order_texts.append(t)
        if n == 0:
            self.roi_boxes.setData([], [])
            self.line_item.setData([], [])
            self.peak_item.setData([], [])
            return
        r = self.params.roi_half
        px = res.peaks[:, 1] + x0
        py = res.peaks[:, 0] + y0
        bx = np.column_stack([px - r, px + r, px + r, px - r, px - r, np.full(n, np.nan)]).ravel()
        by = np.column_stack([py - r, py - r, py + r, py + r, py - r, np.full(n, np.nan)]).ravel()
        self.roi_boxes.setData(bx, by)
        self.peak_item.setData(px + 0.5, py + 0.5)
        ly0, lx0, ly1, lx1 = res.line
        self.line_item.setData([lx0 + x0, lx1 + x0], [ly0 + y0, ly1 + y0])
        for t, order, x, y in zip(self.order_texts, res.orders, px, py):
            t.setText(f"{order:+d}" if order else "0")
            t.setPos(x, y - r)

    def _draw_results(self, res: AnalysisResult, t: float):
        n = len(res.peaks)
        eff = res.efficiencies * 100.0
        self.bar_item.setOpts(x=res.orders, height=eff, width=0.6)
        while len(self._bar_labels) > n:
            self.bar_plot.removeItem(self._bar_labels.pop())
        while len(self._bar_labels) < n:
            lab = pg.TextItem(color="w", anchor=(0.5, 1.0))
            self.bar_plot.addItem(lab)
            self._bar_labels.append(lab)
        for lab, o, e in zip(self._bar_labels, res.orders, eff):
            lab.setText(f"{e:.1f}")
            lab.setPos(o, e)
        if n:
            self.bar_plot.setXRange(res.orders[0] - 0.7, res.orders[-1] + 0.7, padding=0)
            self.bar_plot.setYRange(0, max(eff.max() * 1.2, 1.0), padding=0)
            self.bar_plot.getAxis("bottom").setTicks([[(int(o), str(int(o))) for o in res.orders]])
        self.prof_curve.setData(res.profile)

        self.table.setRowCount(n)
        for i in range(n):
            for j, text in enumerate((f"{res.orders[i]:+d}" if res.orders[i] else "0",
                                      f"{res.intensities[i]:.0f}", f"{eff[i]:.2f} %")):
                item = self.table.item(i, j)
                if item is None:
                    item = QtWidgets.QTableWidgetItem()
                    item.setTextAlignment(QtCore.Qt.AlignCenter)
                    self.table.setItem(i, j, item)
                item.setText(text)

        if n:
            self._history.append((t, dict(zip(res.orders.tolist(), eff.tolist()))))
        times = np.array([h[0] for h in self._history])
        all_orders = sorted({o for _, d in self._history for o in d})
        for o in list(self._hist_curves):
            if o not in all_orders:
                self.hist_plot.removeItem(self._hist_curves.pop(o))
        for o in all_orders:
            if o not in self._hist_curves:
                self._hist_curves[o] = self.hist_plot.plot(
                    pen=pg.mkPen(ORDER_COLORS[o % len(ORDER_COLORS)], width=2), name=f"{o:+d}" if o else "0",
                    connect="finite")
            self._hist_curves[o].setData(times, np.array([d.get(o, np.nan) for _, d in self._history]))

    def clear_history(self):
        self._history.clear()
        for c in self._hist_curves.values():
            self.hist_plot.removeItem(c)
        self._hist_curves.clear()
        self._t0 = time.perf_counter()

    # -------------------------------------------------------------- output
    @staticmethod
    def _rows(res: AnalysisResult, x0: int, y0: int):
        for o, (y, x), inten, eff in zip(res.orders, res.peaks, res.intensities, res.efficiencies):
            yield int(o), int(x + x0), int(y + y0), float(inten), float(eff)

    def save_snapshot(self):
        if self.frame is None:
            return
        frame, res = self.frame.copy(), self.result
        x0, y0 = (0, 0) if self._locked_peaks is not None else self._crop_bounds(frame.shape)[:2]
        default = time.strftime("diffraction_%Y%m%d_%H%M%S.tif")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save snapshot", default, "TIFF (*.tif)")
        if not path:
            return
        path = Path(path).with_suffix(".tif")
        try:
            tifffile.imwrite(path, frame.astype(np.float32))
            with open(path.with_suffix(".csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["order", "x_px", "y_px", "intensity", "efficiency"])
                w.writerows(self._rows(res, x0, y0))
        except OSError as e:
            QtWidgets.QMessageBox.critical(self, "Save snapshot", str(e))
            return
        self.statusBar().showMessage(f"Saved {path.name} and {path.with_suffix('.csv').name}", 5000)

    def toggle_logging(self):
        if self.log_btn.isChecked():
            default = time.strftime("efficiency_log_%Y%m%d_%H%M%S.csv")
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Record to CSV", default, "CSV (*.csv)")
            if not path:
                self.log_btn.setChecked(False)
                return
            try:
                self._log_file = open(path, "w", newline="")
            except OSError as e:
                QtWidgets.QMessageBox.critical(self, "Record to CSV", str(e))
                self.log_btn.setChecked(False)
                return
            self._log_writer = csv.writer(self._log_file)
            self._log_writer.writerow(["timestamp", "order", "x_px", "y_px", "intensity", "efficiency"])
            self.log_btn.setText("Stop recording")
        else:
            if self._log_file is not None:
                self._log_file.close()
            self._log_file = self._log_writer = None
            self.log_btn.setText("Record to CSV…")

    def _write_log(self, res: AnalysisResult, x0: int, y0: int):
        if self._log_writer is None:
            return
        ts = f"{time.time():.3f}"
        for row in self._rows(res, x0, y0):
            self._log_writer.writerow([ts, *row])

    def closeEvent(self, event: QtGui.QCloseEvent):
        self.timer.stop()
        self.disconnect_camera()
        shutdown_sdk()
        super().closeEvent(event)
