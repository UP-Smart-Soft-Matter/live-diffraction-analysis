"""Diffraction-efficiency analysis of a single grayscale frame.

Same method as diffraction_efficiency_class.py in the
diffraction-efficiency-height-retrieval project: find the diffraction orders
as local maxima, sort them along the diffraction axis, sum the pixel values in
a square ROI around each one and normalise by the total.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.ndimage as ndi
from skimage.feature import peak_local_max


@dataclass
class AnalysisParams:
    roi_half: int = 20            # ROI is a (2*roi_half) x (2*roi_half) square
    min_distance: int = 50        # minimum distance between two orders, px
    threshold_rel: float = 0.02   # ignore peaks below this fraction of the brightest
    smooth_sigma: float = 2.0     # Gaussian blur used for peak finding only
    max_orders: int = 0           # keep only the N brightest peaks (0 = all)
    zero_order: str = "center"    # "center" (middle peak) or "brightest"
    subtract_background: bool = False  # subtract the median of the analysed region
    profile_points: int = 1000
    profile_extend: float = 50.0


@dataclass
class AnalysisResult:
    peaks: np.ndarray = field(default_factory=lambda: np.empty((0, 2), int))  # (row, col)
    orders: np.ndarray = field(default_factory=lambda: np.empty(0, int))
    intensities: np.ndarray = field(default_factory=lambda: np.empty(0))
    efficiencies: np.ndarray = field(default_factory=lambda: np.empty(0))
    line: tuple | None = None     # (y0, x0, y1, x1) of the profile line
    profile: np.ndarray = field(default_factory=lambda: np.empty(0))
    background: float = 0.0

    @property
    def ok(self) -> bool:
        return len(self.peaks) > 0


def extend_plot_line(p1, p2, width: float):
    """Extend the segment p1->p2 (both (row, col)) by `width` px at each end."""
    alpha = np.arctan2(p2[0] - p1[0], p2[1] - p1[1])
    dy, dx = np.sin(alpha) * width, np.cos(alpha) * width
    return p1[0] - dy, p1[1] - dx, p2[0] + dy, p2[1] + dx


def find_orders(img: np.ndarray, p: AnalysisParams) -> np.ndarray:
    """Return the (row, col) positions of the orders, sorted along the diffraction axis."""
    none = np.empty((0, 2), int)
    h, w = img.shape
    r = int(p.roi_half)
    work = img.astype(np.float32, copy=False)

    # Large frames are searched on a binned copy (fast), then refined at full resolution.
    s = max(1, min(int(np.ceil(max(h, w) / 512)), int(p.min_distance) // 4))
    if s > 1:
        hh, ww = h // s * s, w // s * s
        if hh == 0 or ww == 0:
            return none
        work = work[:hh, :ww].reshape(hh // s, s, ww // s, s).mean(axis=(1, 3))
    if p.smooth_sigma / s > 0.3:
        work = ndi.gaussian_filter(work, p.smooth_sigma / s)
    work = work - np.median(work)
    peak = float(work.max())
    if peak <= 0:
        return none
    noise = 1.4826 * float(np.median(np.abs(work)))  # robust std of the background
    peaks = peak_local_max(
        work,
        min_distance=max(1, round(p.min_distance / s)),
        threshold_abs=max(p.threshold_rel * peak, 6.0 * noise),  # never pick up pure noise
        exclude_border=False,
        num_peaks=p.max_orders if p.max_orders > 0 else np.inf,
    )
    if s > 1 and len(peaks):
        m = s + 2
        refined = []
        for yc, xc in peaks * s + s // 2:
            ya, xa = max(0, yc - m), max(0, xc - m)
            patch = img[ya:yc + m + 1, xa:xc + m + 1].astype(np.float32)
            if p.smooth_sigma > 0:
                patch = ndi.gaussian_filter(patch, p.smooth_sigma)
            dy, dx = np.unravel_index(np.argmax(patch), patch.shape)
            refined.append((ya + dy, xa + dx))
        peaks = np.array(refined)
    if len(peaks):  # every ROI must lie fully inside the image
        y, x = peaks[:, 0], peaks[:, 1]
        peaks = peaks[(y >= r) & (y <= h - r) & (x >= r) & (x <= w - r)]
    if len(peaks) == 0:
        return none
    sort_col = 1 if np.ptp(peaks[:, 1]) >= np.ptp(peaks[:, 0]) else 0
    return peaks[np.argsort(peaks[:, sort_col])]


def analyze(img: np.ndarray, p: AnalysisParams, peaks: np.ndarray | None = None) -> AnalysisResult:
    """Analyse a 2-D grayscale image. Pass `peaks` to reuse fixed order positions."""
    img = np.asarray(img)
    if peaks is None:
        peaks = find_orders(img, p)
    res = AnalysisResult(peaks=peaks)
    if len(peaks) == 0:
        return res

    if p.subtract_background:
        res.background = float(np.median(img[::4, ::4]))
    bg = res.background

    r = int(p.roi_half)
    h, w = img.shape
    intensities = np.empty(len(peaks))
    for i, (y, x) in enumerate(peaks):
        roi = img[max(0, y - r):min(h, y + r), max(0, x - r):min(w, x + r)]
        intensities[i] = roi.sum(dtype=np.float64) - bg * roi.size
    res.intensities = intensities
    total = intensities.sum()
    res.efficiencies = intensities / total if total > 0 else np.zeros_like(intensities)

    n = len(peaks)
    zero = int(np.argmax(intensities)) if p.zero_order == "brightest" else n // 2
    res.orders = np.arange(n) - zero

    if n >= 2:
        y0, x0, y1, x1 = extend_plot_line(peaks[0], peaks[-1], p.profile_extend)
    else:
        y0 = y1 = float(peaks[0][0])
        x0, x1 = 0.0, float(w - 1)
    res.line = (y0, x0, y1, x1)
    ys = np.linspace(y0, y1, p.profile_points)
    xs = np.linspace(x0, x1, p.profile_points)
    res.profile = ndi.map_coordinates(img, np.vstack((ys, xs)), order=1, mode="nearest").astype(float) - bg
    return res
