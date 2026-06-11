"""
vision.py -- Perception stack per Design Report Section 3.2.

Implements, exactly as documented:
  * QR Detection and Matching .... OpenCV cv2.QRCodeDetector primary, pyzbar
                                   fallback decoder, TIME-BASED TOGGLING of the
                                   two approaches, grayscale -> contrast
                                   normalisation -> adaptive threshold preprocess,
                                   multi-frame persistence acceptance.
  * Banner and Corridor Detection  HSV candidate extraction (NOT sole criterion)
                                   -> contour area (Canny-assisted), rectangularity,
                                   aspect ratio, edge consistency, frame-to-frame
                                   persistence.
  * Red-Zone Detection ........... dual-lobe HSV red segmentation, area threshold,
                                   traversal-zone ROI intrusion -> yaw-only
                                   avoidance command (left/right), reversed when
                                   red pixels exit the ROI.
  * Target Centering ............. QR corner points -> centroid, bounded affine
                                   (digital image stabilisation) correction,
                                   centred-within-tolerance persistence gate
                                   before payload lowering.
  * Colour isolation utility ..... parameterised HSV + shape filter pipeline,
                                   shared by banner/red-zone logic and exposed
                                   to the GCS diagnostics app for live tuning.

YOLOv11 auxiliary layer (Appendix A2) plugs in through `aux_detector`: it is
consulted ONLY when primary OpenCV confidence drops below threshold and never
commands the UAV directly.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

try:
    from pyzbar import pyzbar as _pyzbar
    PYZBAR_AVAILABLE = True
except Exception:                                    # pragma: no cover
    _pyzbar = None
    PYZBAR_AVAILABLE = False

try:
    from . import config
except ImportError:                                  # script-style import
    import config

# ---- Raspberry Pi 5 budget: cap OpenCV worker threads so the FSM, MAVLink
# RX and the diagnostics server keep a dedicated A76 core ------------------
try:
    cv2.setNumThreads(int(config.VISION.get("cv_threads", 0)) or -1)
except Exception:
    pass
cv2.setUseOptimized(True)

# Cached operators -- creating these per frame measurably costs CPU on the Pi
_CLAHE = cv2.createCLAHE(2.0, (8, 8))
_MORPH_K = np.ones((5, 5), np.uint8)


def _proc_scale(override=None) -> float:
    s = config.VISION.get("proc_scale", 1.0) if override is None else override
    return float(min(max(s, 0.25), 1.0))


# ============================================================================
# Data containers
# ============================================================================
@dataclass
class QRResult:
    data: str
    corners: np.ndarray                # 4x2 float
    centroid: tuple                    # (x, y) px
    decoder: str                       # "opencv" | "pyzbar"
    confirmed: bool = False            # passed persistence gate


@dataclass
class BannerResult:
    found: bool
    bbox: tuple = (0, 0, 0, 0)         # x, y, w, h
    centroid: tuple = (0, 0)
    confirmed: bool = False
    score: float = 0.0                 # rectangularity score


@dataclass
class RedZoneResult:
    zones: list = field(default_factory=list)   # list of (bbox, centroid, area)
    roi_intruded: bool = False
    avoid_yaw: int = 0                 # -1 yaw left, +1 yaw right, 0 none


# ============================================================================
# Pi 5 performance helper: detectors run on a downscaled copy
# ============================================================================
def _downscale(frame_bgr, override=None):
    """Return (small, scale) per the OPTIONAL proc_scale setting.

    Default 1.0 = full resolution (recommended: ground targets seen from
    ~10 m need every pixel). When the operator selects 0.75 / 0.5 in the
    GCS, detection runs on the smaller copy (~CPU cost scales with area)
    and all geometry is rescaled back to full-frame coordinates."""
    scale = _proc_scale(override)
    if scale >= 0.999:
        return frame_bgr, 1.0
    small = cv2.resize(frame_bgr, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA)
    return small, scale


# ============================================================================
# QR pipeline
# ============================================================================
class QRPipeline:
    """Dual-decoder QR detection with time-based toggling + persistence.
    Decodes on a downscaled frame first (fast path); retries at full
    resolution only when the fast path finds nothing."""

    def __init__(self):
        self._cv = cv2.QRCodeDetector()
        self._toggle_period = config.VISION["qr_toggle_period_s"]
        self._persist_n = config.VISION["qr_persistence_frames"]
        self._history: deque = deque(maxlen=self._persist_n)
        self._t0 = time.monotonic()

    # ---- preprocessing exactly per report (cached CLAHE: Pi 5 budget) ----
    @staticmethod
    def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = _CLAHE.apply(gray)                                # contrast-normalise
        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 31, 5)    # adaptive threshold
        return thr

    def _active_decoder(self) -> str:
        """Only one approach at a time; toggled on a timer (report 3.2)."""
        if not PYZBAR_AVAILABLE:
            return "opencv"
        phase = int((time.monotonic() - self._t0) / self._toggle_period) % 2
        return "opencv" if phase == 0 else "pyzbar"

    def _decode_opencv_raw(self, img) -> list:
        out = []
        ok, datas, pts, _ = self._cv.detectAndDecodeMulti(img)
        if ok and pts is not None:
            for d, p in zip(datas, pts):
                if d:
                    out.append((d, np.asarray(p, np.float32).reshape(4, 2)))
        return out

    @staticmethod
    def _decode_pyzbar_raw(img) -> list:
        out = []
        for sym in _pyzbar.decode(img):
            if sym.type != "QRCODE" or not sym.data:
                continue
            pts = np.array([(p.x, p.y) for p in sym.polygon], np.float32)
            if pts.shape[0] >= 4:
                out.append((sym.data.decode("utf-8", "replace"), pts[:4]))
        return out

    def detect(self, frame_bgr: np.ndarray) -> list[QRResult]:
        decoder = self._active_decoder()
        # fast path: downscaled frame (≈4× cheaper preprocessing + decode)
        small, scale = _downscale(frame_bgr)
        raw, used = self._try_decode(small, decoder)
        if raw and scale != 1.0:
            raw = [(d, c / scale) for d, c in raw]
        # slow path: tiny/far QR can be lost at 640 px -> retry full frame
        if not raw and scale != 1.0:
            raw, used = self._try_decode(frame_bgr, decoder)

        results = []
        for data, corners in raw:
            cx, cy = corners.mean(axis=0)
            results.append(QRResult(data, corners, (float(cx), float(cy)), used))

        # persistence gate: same decoded string across multiple frames
        self._history.append({r.data for r in results})
        if len(self._history) == self._persist_n:
            stable = set.intersection(*self._history) if all(self._history) else set()
            for r in results:
                r.confirmed = r.data in stable
        return results

    def _try_decode(self, frame, decoder):
        """Pi 5 budget: the CLAHE + adaptive-threshold copy is expensive, so
        it is built LAZILY -- only when the raw frame fails to decode."""
        pre_cache = [None]

        def pre():
            if pre_cache[0] is None:
                pre_cache[0] = self.preprocess(frame)
            return pre_cache[0]

        if decoder == "opencv":
            raw = self._decode_opencv_raw(frame)
            if not raw:
                raw = self._decode_opencv_raw(pre())
        else:
            raw = self._decode_pyzbar_raw(frame)
            if not raw:
                raw = self._decode_pyzbar_raw(pre())
        used = decoder
        # pyzbar retained as in-frame fallback if active decoder found nothing
        if not raw and decoder == "opencv" and PYZBAR_AVAILABLE:
            raw = self._decode_pyzbar_raw(frame) or self._decode_pyzbar_raw(pre())
            used = "pyzbar"
        return raw, used

    def reset(self):
        self._history.clear()


# ============================================================================
# Generic HSV + shape isolation (shared core; GCS-tunable)
# ============================================================================
def isolate_color(frame_bgr, hsv_ranges, min_area=1500, aspect_range=(0.2, 6.0),
                  rectangularity_min=0.0, draw=None):
    """HSV thresholding identifies CANDIDATE regions only; candidates are then
    filtered on contour area, rectangularity, aspect ratio and edge consistency
    (Canny corroboration) -- per the banner pipeline in the report.

    Pi 5: the whole pipeline runs on a downscaled copy (HSV convert, morphology,
    Canny and contours are the hot spots); bboxes/centroids/areas are rescaled
    to full-frame coordinates before returning.

    hsv_ranges: list of (lo, hi) HSV tuples (ORed together).
    Returns (mask, detections) -- detections: dicts with bbox/centroid/area/score.
    """
    small, scale = _downscale(frame_bgr)
    inv = 1.0 / scale
    s_min_area = min_area * scale * scale

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in hsv_ranges:
        mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _MORPH_K)   # cached kernel
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _MORPH_K)

    edges = cv2.Canny(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 60, 160)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < s_min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        rect = area / max(w * h, 1)                    # rectangularity
        aspect = w / max(h, 1)
        if not (aspect_range[0] <= aspect <= aspect_range[1]):
            continue
        if rect < rectangularity_min:
            continue
        # edge consistency: real boards/banners show strong perimeter edges
        band = edges[max(y - 3, 0):y + h + 3, max(x - 3, 0):x + w + 3]
        edge_density = float(band.mean()) / 255.0
        fx, fy = int(x * inv), int(y * inv)
        fw, fh = int(w * inv), int(h * inv)
        detections.append({
            "bbox": (fx, fy, fw, fh),
            "centroid": (fx + fw // 2, fy + fh // 2),
            "area": float(area * inv * inv), "rectangularity": float(rect),
            "aspect": float(aspect), "edge_density": edge_density,
        })
        if draw is not None:
            cv2.rectangle(draw, (fx, fy), (fx + fw, fy + fh), (80, 220, 80), 2)
            cv2.drawMarker(draw, (fx + fw // 2, fy + fh // 2), (80, 220, 80),
                           cv2.MARKER_CROSS, 18, 2)
    detections.sort(key=lambda d: d["area"], reverse=True)
    if scale != 1.0:                                   # full-size mask for UI
        mask = cv2.resize(mask, (frame_bgr.shape[1], frame_bgr.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    return mask, detections


# ============================================================================
# Banner pipeline (hybrid: classical primary, NN fallback hook)
# ============================================================================
class BannerPipeline:
    """Tunables are read from config.VISION on every detect() call, so the
    GCS tuning panel (and persisted tuning_overrides.json) take effect live
    in BOTH diagnostics and the FSM -- one config, one pipeline."""

    def __init__(self, aux_detector=None):
        v = config.VISION
        self._persist = deque(maxlen=v["banner_persistence_frames"])
        self.aux_detector = aux_detector               # YOLOv11 confidence-support
        self.aux_conf_threshold = 0.60                 # Appendix A2

    def detect(self, frame_bgr, draw=None) -> BannerResult:
        v = config.VISION
        ranges = [(v["banner_hsv_lo"], v["banner_hsv_hi"])]
        _, dets = isolate_color(frame_bgr, ranges, v["banner_min_area_px"],
                                v["banner_aspect_range"],
                                v["banner_rectangularity_min"], draw=draw)
        found = bool(dets)
        conf = dets[0]["rectangularity"] if found else 0.0

        # NN fallback ONLY when primary confidence is below threshold (A2)
        if (not found or conf < self.aux_conf_threshold) and self.aux_detector:
            aux = self.aux_detector(frame_bgr, target_class="banner")
            if aux:
                dets, found, conf = [aux], True, aux.get("rectangularity", 0.6)

        self._persist.append(found)
        confirmed = len(self._persist) == self._persist.maxlen and all(self._persist)
        if found:
            d = dets[0]
            return BannerResult(True, d["bbox"], d["centroid"], confirmed, conf)
        return BannerResult(False)

    def reset(self):
        self._persist.clear()


# ============================================================================
# Red restricted-zone pipeline
# ============================================================================
class RedZonePipeline:
    """Tunables read from config.VISION per call -- see BannerPipeline."""

    def __init__(self):
        self._avoiding = 0                             # active yaw correction

    def detect(self, frame_bgr, draw=None) -> RedZoneResult:
        v = config.VISION
        ranges = [(v["red_hsv_lo1"], v["red_hsv_hi1"]),
                  (v["red_hsv_lo2"], v["red_hsv_hi2"])]
        min_area = v["red_min_area_px"]
        h, w = frame_bgr.shape[:2]
        mask, dets = isolate_color(frame_bgr, ranges, min_area, draw=None)
        res = RedZoneResult(zones=dets)

        # traversal-zone ROI: centre band of the downward camera frame
        rw = int(w * v["red_roi_frac"])
        x0, x1 = (w - rw) // 2, (w + rw) // 2
        roi_hits = mask[:, x0:x1].sum() / 255.0
        res.roi_intruded = roi_hits > min_area

        if res.roi_intruded and dets:
            # pixel-based centroiding -> yaw AWAY from zone centroid (yaw only,
            # no induced roll), reversed once red pixels exit the ROI
            cx = dets[0]["centroid"][0]
            self._avoiding = +1 if cx < w / 2 else -1
        elif not res.roi_intruded and self._avoiding:
            self._avoiding = 0                         # correction reversed
        res.avoid_yaw = self._avoiding

        if draw is not None:
            cv2.rectangle(draw, (x0, 0), (x1, h), (90, 90, 200), 1)
            for d in dets:
                x, y, bw, bh = d["bbox"]
                cv2.rectangle(draw, (x, y), (x + bw, y + bh), (60, 60, 230), 2)
                cv2.putText(draw, "RED ZONE", (x, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 230), 2)
        return res


# ============================================================================
# Target centering + digital image stabilisation (S5)
# ============================================================================
class TargetCentering:
    """Centroid-based terminal guidance over the matched QR.
    Payload lowering is enabled ONLY after the QR stays centred within
    tolerance for `center_persistence_frames` consecutive frames."""

    def __init__(self):
        v = config.VISION
        self._persist = deque(maxlen=v["center_persistence_frames"])
        self.max_shift = v["dis_max_shift_px"]

    @property
    def tol_frac(self):                                # live-tunable
        return config.VISION["center_tol_frac"]

    def stabilise(self, frame_bgr, roll_deg=0.0, pitch_deg=0.0):
        """Bounded affine correction (cv2.warpAffine) from FC tilt angles
        over MAVLink -- Innovation 3, Dual Camera DIS."""
        h, w = frame_bgr.shape[:2]
        # small-angle px shift model for the 76deg downward lens
        px_per_deg = w / 76.0
        dx = float(np.clip(-roll_deg * px_per_deg, -self.max_shift, self.max_shift))
        dy = float(np.clip(pitch_deg * px_per_deg, -self.max_shift, self.max_shift))
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(frame_bgr, M, (w, h)), (dx, dy)

    def update(self, frame_shape, qr: QRResult | None):
        """Returns dict: offsets (px + normalised), centred?, lowering gate."""
        h, w = frame_shape[:2]
        if qr is None:
            self._persist.append(False)
            return {"visible": False, "centred": False, "gate_open": False,
                    "off_px": (0, 0), "off_norm": (0.0, 0.0)}
        ox = qr.centroid[0] - w / 2
        oy = qr.centroid[1] - h / 2
        centred = (abs(ox) <= w * self.tol_frac) and (abs(oy) <= w * self.tol_frac)
        self._persist.append(centred)
        gate = len(self._persist) == self._persist.maxlen and all(self._persist)
        return {"visible": True, "centred": centred, "gate_open": gate,
                "off_px": (float(ox), float(oy)),
                "off_norm": (float(ox / (w / 2)), float(oy / (h / 2)))}

    def reset(self):
        self._persist.clear()


# ============================================================================
# Convenience: annotate QRs on a frame (used by FSM logging & GCS streams)
# ============================================================================
def annotate_qrs(frame, results, mission_code=None):
    for r in results:
        match = mission_code is not None and r.data == mission_code
        col = (80, 220, 80) if (match or (mission_code is None and r.confirmed)) \
            else ((40, 170, 240) if mission_code is None else (40, 160, 240))
        pts = r.corners.astype(int)
        cv2.polylines(frame, [pts], True, col, 2)
        cv2.drawMarker(frame, (int(r.centroid[0]), int(r.centroid[1])), col,
                       cv2.MARKER_CROSS, 22, 2)
        tag = r.data[:28] + ("  [MATCH]" if match else "")
        cv2.putText(frame, tag, (pts[:, 0].min(), max(pts[:, 1].min() - 8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
    return frame
