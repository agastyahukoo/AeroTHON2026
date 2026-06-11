"""
detlog.py -- Timestamped detection logging (flight-line evidence trail).

Every confirmed detection from the vision stack is appended to a daily JSONL
file in onboard/logs/. One JSON object per line, e.g. a QR entry:

  {"ts": "2026-06-12T10:41:03.214+05:30", "t_mono": 1234.56,
   "type": "qr", "data": "DELIVERY:ZONE-B", "decoder": "opencv",
   "px": [641.2, 388.7], "gps": {"lat": 19.107612, "lon": 72.837041,
   "alt_m": 9.98}, "state": "S4"}

Rules:
  * QR entries always carry decoded TEXT + pixel LOCATION + GPS fix (when a
    MAVLink position is available -- the FSM injects it; bench diagnostics
    log gps: null).
  * banner / redzone entries carry bbox + centroid (+ GPS when flying).
  * Writes are line-buffered, fsync'd on every Nth line, and serialized by a
    lock -- safe across the FSM thread and the diagnostics server thread.
  * Duplicate suppression: the same (type, data) is not re-logged within
    `dedup_window_s`, so a QR held in frame logs once, not 15x per second.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone

try:
    from . import config
except ImportError:
    import config

_FSYNC_EVERY = 5


class DetectionLogger:
    def __init__(self, dedup_window_s: float = 3.0):
        os.makedirs(config.LOGGING["dir"], exist_ok=True)
        self._lock = threading.Lock()
        self._fh = None
        self._path = None
        self._lines = 0
        self._dedup: dict[tuple, float] = {}
        self.dedup_window_s = dedup_window_s

    # ------------------------------------------------------------------ io
    def _ensure_file(self):
        path = os.path.join(
            config.LOGGING["dir"],
            datetime.now().strftime(config.LOGGING["detections_file"]))
        if path != self._path:
            if self._fh:
                self._fh.close()
            self._fh = open(path, "a", buffering=1)        # line-buffered
            self._path = path
        return self._fh

    @property
    def path(self) -> str | None:
        with self._lock:
            self._ensure_file()
            return self._path

    # ----------------------------------------------------------------- log
    def log(self, dtype: str, data: str = "", px=None, bbox=None,
            gps=None, decoder=None, state=None, extra=None) -> bool:
        """Returns True if written, False if suppressed as a duplicate."""
        now = time.monotonic()
        key = (dtype, data)
        with self._lock:
            last = self._dedup.get(key, 0.0)
            if now - last < self.dedup_window_s:
                return False
            self._dedup[key] = now
            entry = {
                "ts": datetime.now(timezone.utc).astimezone().isoformat(
                    timespec="milliseconds"),
                "t_mono": round(now, 3),
                "type": dtype,
            }
            if data:
                entry["data"] = data
            if decoder:
                entry["decoder"] = decoder
            if px is not None:
                entry["px"] = [round(float(px[0]), 1), round(float(px[1]), 1)]
            if bbox is not None:
                entry["bbox"] = [int(v) for v in bbox]
            entry["gps"] = ({"lat": round(gps[0], 7), "lon": round(gps[1], 7),
                             "alt_m": round(gps[2], 2)}
                            if gps is not None else None)
            if state:
                entry["state"] = state
            if extra:
                entry.update(extra)
            fh = self._ensure_file()
            fh.write(json.dumps(entry) + "\n")
            self._lines += 1
            if self._lines % _FSYNC_EVERY == 0:
                os.fsync(fh.fileno())
            return True

    # ---------------------------------------------------------------- read
    def tail(self, n: int = 50) -> list:
        with self._lock:
            self._ensure_file()
            path = self._path
        try:
            with open(path) as f:
                lines = f.readlines()[-n:]
            return [json.loads(x) for x in lines if x.strip()]
        except Exception:
            return []


# Module-level singleton: FSM and diagnostics server share one logger
LOGGER = DetectionLogger()
