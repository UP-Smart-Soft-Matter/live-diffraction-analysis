"""Camera backends: Thorlabs scientific cameras (TSI SDK) and a simulator.

The Thorlabs backend uses the official `thorlabs_tsi_sdk` Python package that
ships with ThorCam (Zelux, Kiralux, Quantalux, CS-series ... cameras). See the
README for how to install it.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import numpy as np


class CameraError(RuntimeError):
    pass


class Camera:
    """Minimal interface the GUI needs. Frames are 2-D uint16 arrays."""

    name = "camera"
    bit_depth = 8
    exposure_range_ms = (0.01, 1000.0)
    gain_range = (0, 0)  # (0, 0) = no gain control

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...

    def read(self) -> np.ndarray | None:
        """Return the next frame, or None if none arrived within the poll timeout."""
        raise NotImplementedError

    def get_exposure_ms(self) -> float:
        raise NotImplementedError

    def set_exposure_ms(self, value: float) -> None:
        raise NotImplementedError

    def get_gain(self) -> int:
        return 0

    def set_gain(self, value: int) -> None: ...

    @property
    def max_value(self) -> int:
        return (1 << self.bit_depth) - 1


# --------------------------------------------------------------------------
# Thorlabs TSI SDK
# --------------------------------------------------------------------------

_DLL_NAME = "thorlabs_tsi_camera_sdk.dll"
_sdk = None
_sdk_lock = threading.Lock()


def _dll_candidates() -> list[Path]:
    here = Path(__file__).resolve().parent.parent
    out = []
    if os.environ.get("THORLABS_TSI_DLL_DIR"):
        out.append(Path(os.environ["THORLABS_TSI_DLL_DIR"]))
    out.append(here / "dlls")
    for pf in (os.environ.get("ProgramFiles"), os.environ.get("ProgramW6432")):
        if not pf:
            continue
        base = Path(pf) / "Thorlabs" / "Scientific Imaging"
        out += [
            base / "ThorCam",
            base / "Scientific Camera Support" / "Scientific Camera Interfaces"
            / "SDK" / "Native Toolkit" / "dlls" / "Native_64_lib",
        ]
    return out


def _setup_dll_path() -> Path | None:
    for d in _dll_candidates():
        if (d / _DLL_NAME).is_file():
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(d))
            return d
    return None


def _get_sdk():
    """Create the (singleton) TLCameraSDK on first use."""
    global _sdk
    with _sdk_lock:
        if _sdk is not None:
            return _sdk
        if sys.platform == "win32":
            dll_dir = _setup_dll_path()
        else:
            dll_dir = None
        try:
            from thorlabs_tsi_sdk.tl_camera import TLCameraSDK
        except ImportError as e:
            raise CameraError(
                "The Thorlabs Python package 'thorlabs_tsi_sdk' is not installed.\n"
                "Install ThorCam, then pip-install the SDK zip from its "
                "'Scientific Camera Interfaces' folder (see README)."
            ) from e
        try:
            _sdk = TLCameraSDK()
        except Exception as e:
            hint = "" if dll_dir else (
                f"\n{_DLL_NAME} was not found. Install ThorCam, copy the SDK DLLs "
                "into the 'dlls' folder next to main.py, or set THORLABS_TSI_DLL_DIR."
            )
            raise CameraError(f"Could not start the Thorlabs camera SDK: {e}{hint}") from e
        return _sdk


def list_thorlabs_cameras() -> list[str]:
    """Serial numbers of the connected Thorlabs cameras. Raises CameraError."""
    try:
        return list(_get_sdk().discover_available_cameras())
    except CameraError:
        raise
    except Exception as e:
        raise CameraError(f"Camera discovery failed: {e}") from e


def shutdown_sdk() -> None:
    global _sdk
    with _sdk_lock:
        if _sdk is not None:
            try:
                _sdk.dispose()
            except Exception:
                pass
            _sdk = None


class ThorlabsCamera(Camera):
    def __init__(self, serial: str):
        sdk = _get_sdk()
        try:
            self._cam = sdk.open_camera(serial)
        except Exception as e:
            raise CameraError(f"Could not open camera {serial}: {e}") from e
        cam = self._cam
        self._lock = threading.Lock()
        self._armed = False
        self.name = f"{cam.model} ({serial})"
        self.bit_depth = int(cam.bit_depth)
        rng = cam.exposure_time_range_us
        self.exposure_range_ms = (rng.min / 1000.0, rng.max / 1000.0)
        try:
            g = cam.gain_range
            self.gain_range = (int(g.min), int(g.max))
        except Exception:
            self.gain_range = (0, 0)
        cam.frames_per_trigger_zero_for_unlimited = 0  # continuous
        cam.image_poll_timeout_ms = 200

    def start(self) -> None:
        with self._lock:
            if not self._armed:
                self._cam.arm(2)
                self._cam.issue_software_trigger()
                self._armed = True

    def stop(self) -> None:
        with self._lock:
            if self._armed:
                self._cam.disarm()
                self._armed = False

    def close(self) -> None:
        try:
            self.stop()
        finally:
            with self._lock:
                self._cam.dispose()

    def read(self) -> np.ndarray | None:
        with self._lock:
            if not self._armed:
                return None
            frame = self._cam.get_pending_frame_or_null()
            if frame is None:
                return None
            # the SDK reuses the buffer, so copy it
            return np.array(frame.image_buffer, dtype=np.uint16, copy=True)

    def get_exposure_ms(self) -> float:
        with self._lock:
            return self._cam.exposure_time_us / 1000.0

    def set_exposure_ms(self, value: float) -> None:
        lo, hi = self.exposure_range_ms
        us = int(round(min(max(value, lo), hi) * 1000))
        with self._lock:
            self._cam.exposure_time_us = us

    def get_gain(self) -> int:
        if self.gain_range[1] <= 0:
            return 0
        with self._lock:
            return int(self._cam.gain)

    def set_gain(self, value: int) -> None:
        if self.gain_range[1] <= 0:
            return
        lo, hi = self.gain_range
        with self._lock:
            self._cam.gain = int(min(max(value, lo), hi))


# --------------------------------------------------------------------------
# Simulator
# --------------------------------------------------------------------------

class SimulatedCamera(Camera):
    """Synthetic diffraction pattern of a sinusoidal phase grating whose
    modulation depth drifts slowly, so the efficiencies change over time.
    If `image` is given, that image is replayed with noise instead."""

    name = "Simulated camera"
    bit_depth = 10
    exposure_range_ms = (0.1, 200.0)
    gain_range = (0, 0)

    def __init__(self, image: np.ndarray | None = None, fps: float = 20.0):
        self._exposure = 10.0
        self._period = 1.0 / fps
        self._running = False
        self._t0 = time.perf_counter()
        self._rng = np.random.default_rng()
        self._image = None
        if image is not None:
            img = np.asarray(image, dtype=np.float32)
            if img.ndim == 3:
                img = img[..., :3].mean(axis=2)
            self._image = img / max(float(img.max()), 1.0)
        else:
            h, w = 1080, 1440
            self._orders = np.arange(-2, 3)
            self._shape = (h, w)
            k = 40  # spots are drawn as small patches
            yy, xx = np.mgrid[-k:k + 1, -k:k + 1].astype(np.float32)
            self._spot = np.exp(-(xx ** 2 + yy ** 2) / (2 * 7.0 ** 2))
            self._slices = []
            for m in self._orders:
                cx, cy = w // 2 + m * 110, h // 2 + m * 4
                self._slices.append((slice(cy - k, cy + k + 1), slice(cx - k, cx + k + 1)))

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def read(self) -> np.ndarray | None:
        time.sleep(self._period)
        if not self._running:
            return None
        t = time.perf_counter() - self._t0
        if self._image is not None:
            frame = self._image * (1 + 0.03 * np.sin(t)) * 60.0
        else:
            from scipy.special import jv
            phi = 1.65 + 0.35 * np.sin(2 * np.pi * t / 20.0)
            eff = jv(self._orders, phi) ** 2
            frame = np.zeros(self._shape, np.float32)
            for e, sl in zip(eff, self._slices):
                frame[sl] += e * 140.0 * self._spot
        frame = frame * self._exposure + 12.0
        frame = frame + 2.0 * self._rng.standard_normal(frame.shape, dtype=np.float32)
        return np.clip(frame, 0, self.max_value).astype(np.uint16)

    def get_exposure_ms(self) -> float:
        return self._exposure

    def set_exposure_ms(self, value: float) -> None:
        lo, hi = self.exposure_range_ms
        self._exposure = float(min(max(value, lo), hi))
