#!/usr/bin/env python3
"""
demo_server.py -- AT2026RC67 ground diagnostics bridge.

This file lives OUTSIDE the onboard/ folder on purpose: it contains zero
mission logic. It only imports the exact flight modules from onboard/ and
exposes them to the GCS web application (webapp.html) over HTTP, so every
demo on screen is running the same code that flies.

    python3 demo_server.py            # serves http://localhost:8742
    python3 demo_server.py --port N --cam 0

Endpoints
  GET  /                       webapp.html
  GET  /stream                 MJPEG of the active vision mode
  GET  /api/specs              airframe spec sheet (onboard config.SPECS)
  GET  /api/status             link/camera/decoder/runtime status
  POST /api/mode               {"mode": fpv|qr|multiqr|isolate|banner|lowering}
  POST /api/source             {"type": camera|image|video|pause, ...}
  POST /api/scale              {"scale": 1.0|0.75|0.5}  processing downscale
  GET  /api/detlog             timestamped detection log (tail)
  GET  /api/detlog/download    full JSONL detection log file
  GET  /api/qr                 decoded QR list + mission code + matches
  POST /api/qr/mission         {"code": "..."} | {"auto": true} | {"clear":true}
  POST /api/isolate            HSV ranges + shape-filter params (live tuning)
  GET  /api/lowering           payload-delivery demo state
  POST /api/lowering           {"action": start|reset}
  GET  /api/fsm  POST /api/fsm FSM dry-run snapshot / {"action": start|stop}
  GET/POST /api/geofence       mission_geofence.json read / write
  GET  /api/export             ZIP of onboard/ (ready for the Pi 5)
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import sys
import tempfile
import threading
import time
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "onboard"))

import config                                    # noqa: E402  (onboard/)
import vision                                    # noqa: E402
from geofence import GeofenceManager             # noqa: E402
from winch import Winch                          # noqa: E402
from fsm import MissionFSM                       # noqa: E402
from detlog import LOGGER as DETLOG              # noqa: E402

# ---- Raspberry Pi 5 runtime tuning ----------------------------------------
cv2.setUseOptimized(True)                        # NEON/IPP fast paths
cv2.setNumThreads(config.VISION.get("cv_threads", 3))   # leave a core free

try:                                             # CSI cameras via libcamera —
    from picamera2 import Picamera2              # far cheaper than V4L2/FFmpeg
    PICAM_AVAILABLE = True
except Exception:
    Picamera2 = None
    PICAM_AVAILABLE = False


# ============================================================== video source
class VideoSource:
    """camera | image | video -- one REAL frame provider. There is no
    synthetic scene: with no input attached the stream shows a NO INPUT
    status card, which by construction cannot produce a detection.

    Pi 5 budget: the camera kind runs a grabber thread that keeps only the
    newest frame, so pipelines never process stale V4L2 buffers and the
    processing loop never blocks on capture."""

    def __init__(self, cam_index=0):
        self.lock = threading.Lock()
        self.kind = "camera"
        self.cam_index = cam_index
        self._cap = None
        self._picam = None
        self._still = None
        self._held = None                      # last frame, shown while paused
        self.meta = {"name": f"camera {cam_index}", "paused": False}
        self._cam_latest = None
        self._open_camera()
        if self.kind == "none":
            self.meta = {"name": "no input -- select camera / image / video",
                         "paused": False}

    def _open_camera(self):
        self._close_camera()
        # Pi 5: Picamera2/libcamera for CSI modules (IMX708) — hardware ISP,
        # no FFmpeg copy chain, dramatically lower CPU than cv2.VideoCapture
        if PICAM_AVAILABLE:
            try:
                cam = Picamera2(self.cam_index)
                cam.configure(cam.create_video_configuration(
                    main={"size": (1280, 720), "format": "RGB888"}))
                cam.start()
                self._picam = cam
                return
            except Exception:
                self._picam = None
        self._cap = cv2.VideoCapture(self.cam_index)
        if self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # always-fresh frames
            self._start_grabber()
        else:
            self._cap.release()
            self._cap = None
            self.kind = "none"

    def _close_camera(self):
        if getattr(self, "_picam", None) is not None:
            try:
                self._picam.stop()
                self._picam.close()
            except Exception:
                pass
            self._picam = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def set_camera(self, index) -> bool:
        with self.lock:
            self.kind, self.cam_index = "camera", index
            self._still = None                 # drop any previous image
            self._held = None                  # drop any previous video frame
            self._cam_latest = None            # never show a stale camera frame
            self._open_camera()
            ok = self.kind == "camera"
            self.meta = {"name": f"camera {index}" if ok else
                         f"no input -- camera {index} failed to open",
                         "paused": False}
            return ok

    def set_image(self, bgr, name="image"):
        with self.lock:
            self._close_camera()               # release device + stop grabber
            self._cam_latest = None            # no stale camera frame possible
            self._held = None
            self.kind, self._still = "image", bgr
            h, w = bgr.shape[:2]
            self.meta = {"name": name, "res": f"{w}x{h}", "paused": False}

    def set_video(self, path, name="video") -> bool:
        with self.lock:
            self._close_camera()
            self._cam_latest = None
            self._still = None
            self._cap = cv2.VideoCapture(path)
            ok = self._cap.isOpened()
            if not ok:
                self._cap.release()
                self._cap = None
            self.kind = "video" if ok else "none"
            self.meta = {"name": name if ok else
                         "no input -- video failed to open",
                         "res": f"{int(self._cap.get(3))}x{int(self._cap.get(4))}"
                         if ok else "",
                         "fps": round(self._cap.get(cv2.CAP_PROP_FPS) or 0, 1)
                         if ok else 0,
                         "paused": False}
            self._held = None
            return ok

    def _start_grabber(self):
        if getattr(self, "_grab_on", False):
            return
        self._grab_on = True
        threading.Thread(target=self._grab_loop, daemon=True).start()

    def _grab_loop(self):
        while True:
            with self.lock:
                cap = self._cap if self.kind == "camera" else None
            if cap is None:
                self._grab_on = False
                return
            ok, f = cap.read()
            if ok:
                self._cam_latest = f
            else:
                time.sleep(0.05)

    def set_paused(self, paused: bool):
        with self.lock:
            self.meta["paused"] = bool(paused)

    def info(self):
        with self.lock:
            return {"kind": self.kind, **self.meta}

    # NO INPUT status card -- a static frame that cannot ever contain a
    # detectable target. Synthetic detection scenes are deliberately absent:
    # every result shown by the GCS comes from a real camera, image or video.
    _CARD = None

    def _no_input(self):
        if VideoSource._CARD is None:
            f = np.zeros((720, 1280, 3), np.uint8)
            cv2.rectangle(f, (8, 8), (1271, 711), (38, 41, 45), 1)
            cv2.putText(f, "NO INPUT", (492, 340),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (138, 144, 151), 2)
            cv2.putText(f, "select camera / image / video in the GCS",
                        (380, 396), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (80, 86, 93), 1)
            VideoSource._CARD = f
        return VideoSource._CARD.copy()

    def frame(self):
        with self.lock:
            if self.kind == "image" and self._still is not None:
                return self._still.copy()
            if self.kind == "camera" and self._picam is not None:
                try:
                    return self._picam.capture_array()      # RGB888 == BGR order
                except Exception:
                    pass
            if self.kind == "camera" and self._cap is not None:
                if self._cam_latest is not None:
                    return self._cam_latest.copy()     # grabber thread output
            if self.kind == "video" and self._cap is not None:
                if self.meta.get("paused") and self._held is not None:
                    return self._held.copy()           # hold frame for stepping
                ok, frm = self._cap.read()
                if not ok:                             # loop video
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frm = self._cap.read()
                if ok:
                    self._held = frm
                    return frm
            return self._no_input()


# ============================================================ GCS settings
SETTINGS_FILE = os.path.join(ROOT, "gcs_settings.json")
SETTINGS_DEFAULTS = {"operator": "", "default_cam": 0,
                     "stream_quality": 70, "proc_scale": 1.0}


def load_settings() -> dict:
    s = dict(SETTINGS_DEFAULTS)
    try:
        with open(SETTINGS_FILE) as f:
            s.update({k: v for k, v in json.load(f).items()
                      if k in SETTINGS_DEFAULTS})
    except Exception:
        pass
    return s


def apply_settings(s: dict):
    config.VISION["stream_quality"] = int(s.get("stream_quality", 70))
    config.VISION["proc_scale"] = float(s.get("proc_scale", 1.0))


def save_settings(s: dict) -> dict:
    cur = load_settings()
    cur.update({k: v for k, v in s.items() if k in SETTINGS_DEFAULTS})
    with open(SETTINGS_FILE, "w") as f:
        json.dump(cur, f, indent=2)
    apply_settings(cur)
    return cur


# ====================================================== diagnostics core
class Diagnostics:
    """Owns the onboard pipelines and the per-mode processing loop."""

    def __init__(self, cam_index):
        self.src = VideoSource(cam_index)
        self.mode = "fpv"
        self.qr = vision.QRPipeline()
        self.banner = vision.BannerPipeline()
        self.redzone = vision.RedZonePipeline()
        self.center = vision.TargetCentering()
        self.fence = GeofenceManager()

        self.mission_code: str | None = None
        self.last_qr: list = []
        self.qr_log: list = []

        # live-tunable isolation parameters (colour-picker driven)
        self.iso = {"ranges": [[[0, 110, 70], [10, 255, 255]],
                               [[170, 110, 70], [180, 255, 255]]],
                    "min_area": 2500, "aspect": [0.2, 6.0], "rect_min": 0.55,
                    "label": "RED ZONE"}

        # payload-lowering demo
        self.low = {"phase": "STANDBY", "alt_m": 10.0, "centred": False,
                    "gate": False, "off": (0.0, 0.0), "t0": None}
        self.winch = Winch(simulate=True, time_scale=12.0)

        # FSM dry run (sim backends, frames from the active source)
        self.fsm: MissionFSM | None = None

        self._jpeg = None
        self._jpeg_lock = threading.Lock()
        self.clients = 0                       # active /stream viewers
        self._frame_i = 0
        self._fps = 0.0
        threading.Thread(target=self._loop, daemon=True).start()

    # ----------------------------------------------------------- main loop
    def _loop(self):
        self._frame_i = 0
        self._fps = 0.0
        while True:
            t0 = time.monotonic()
            # read every tick so Settings changes apply live
            sw = config.VISION.get("stream_width", 960)
            sq = int(config.VISION.get("stream_quality", 70))
            # idle throttle: with no stream viewers, tick at 2 fps to keep
            # demo state machines alive while freeing the CPU on the Pi
            idle = self.clients == 0
            frame = self.src.frame()
            self._frame_i += 1
            try:
                frame = getattr(self, f"_m_{self.mode}")(frame)
            except Exception as e:                  # never kill the stream
                cv2.putText(frame, f"pipeline error: {e}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            if not idle:
                if frame.shape[1] > sw:             # encode small: ~2.5x less
                    h = int(frame.shape[0] * sw / frame.shape[1])
                    frame = cv2.resize(frame, (sw, h),
                                       interpolation=cv2.INTER_AREA)
                fh, fw = frame.shape[:2]            # FPS overlay, every mode
                tag = (f"{self._fps:4.1f} FPS  "
                       f"x{config.VISION.get('proc_scale', 1.0):.2f}")
                cv2.putText(frame, tag, (fw - 196, fh - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
                cv2.putText(frame, tag, (fw - 196, fh - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (201, 209, 217), 1)
                ok, jpg = cv2.imencode(".jpg", frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, sq])
                if ok:
                    with self._jpeg_lock:
                        self._jpeg = jpg.tobytes()
            dt = time.monotonic() - t0
            period = 0.5 if idle else 1.0 / config.VISION["target_fps"]
            time.sleep(max(0, period - dt))
            cycle = time.monotonic() - t0          # true loop period
            self._fps = round(0.85 * self._fps +
                              0.15 * (1.0 / max(cycle, 1e-3)), 1)

    def jpeg(self):
        with self._jpeg_lock:
            return self._jpeg

    def snapshot_jpeg(self):
        """On-demand full-size capture of the ACTIVE pipeline output --
        works even while the stream loop is idle (no viewers)."""
        frame = self.src.frame()
        try:
            frame = getattr(self, f"_m_{self.mode}")(frame)
        except Exception:
            pass
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return jpg.tobytes() if ok else (self.jpeg() or b"")

    MODES = ("fpv", "qr", "multiqr", "isolate", "banner", "lowering", "fsm")

    def set_mode(self, mode):
        self.mode = mode if mode in self.MODES else "fpv"
        self.qr.reset()
        self.banner.reset()
        self.center.reset()

    # ----------------------------------------------------------- per mode
    def _hud(self, frame, lines):
        y = 26
        for txt in lines:
            cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62, (0, 0, 0), 4)
            cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62, (210, 235, 245), 1)
            y += 26
        return frame

    def _m_fpv(self, frame):
        return self._hud(frame, [f"FM1 FPV | {self.src.kind.upper()} | "
                                 f"{self._fps:.0f} fps | "
                                 f"{time.strftime('%H:%M:%S')}"])

    def _record_qrs(self, results):
        self.last_qr = [{"data": r.data, "decoder": r.decoder,
                         "confirmed": r.confirmed,
                         "match": (self.mission_code is not None
                                   and r.data == self.mission_code)}
                        for r in results]
        for r in results:
            if r.confirmed and (not self.qr_log or
                                self.qr_log[-1]["data"] != r.data):
                self.qr_log.append({"t": time.strftime("%H:%M:%S"),
                                    "data": r.data, "decoder": r.decoder})
                self.qr_log = self.qr_log[-30:]
            if r.confirmed:                     # timestamped file evidence
                DETLOG.log("qr", r.data, px=r.centroid, gps=self._gps(),
                           decoder=r.decoder, state="DIAG",
                           extra={"match": (self.mission_code is not None
                                            and r.data == self.mission_code)})

    def _m_qr(self, frame):
        results = self.qr.detect(frame)
        self._record_qrs(results)
        vision.annotate_qrs(frame, results)
        dec = "opencv" if not results else results[0].decoder
        return self._hud(frame, [
            f"QR DECODE | active decoder: {dec} (time-toggled)",
            f"persistence gate: {config.VISION['qr_persistence_frames']} frames"])

    def _m_multiqr(self, frame):
        results = self.qr.detect(frame)
        # auto-capture: first confirmed QR becomes the mission delivery code
        if self.mission_code is None:
            for r in results:
                if r.confirmed:
                    self.mission_code = r.data
                    break
        self._record_qrs(results)
        vision.annotate_qrs(frame, results, mission_code=self.mission_code)
        return self._hud(frame, [
            "TARGET IDENTIFICATION (S4)",
            f"mission code: {self.mission_code or '-- scan start QR --'}",
            "green = MATCH, amber = non-matching QR"])

    def _m_isolate(self, frame):
        ranges = [(tuple(lo), tuple(hi)) for lo, hi in self.iso["ranges"]]
        mask, dets = vision.isolate_color(
            frame, ranges, self.iso["min_area"],
            tuple(self.iso["aspect"]), self.iso["rect_min"], draw=frame)
        small = cv2.cvtColor(cv2.resize(mask, (320, 180)), cv2.COLOR_GRAY2BGR)
        frame[frame.shape[0] - 188:frame.shape[0] - 8,
              frame.shape[1] - 328:frame.shape[1] - 8] = small
        for d in dets:
            x, y, w, h = d["bbox"]
            cv2.putText(frame, f"{self.iso['label']} r={d['rectangularity']:.2f}"
                        f" ar={d['aspect']:.2f}", (x, y + h + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 220, 80), 2)
        return self._hud(frame, [
            "COLOUR ISOLATION | HSV candidates -> shape filter",
            f"accepted: {len(dets)}  (area>{self.iso['min_area']}px, "
            f"rect>{self.iso['rect_min']}, AR {self.iso['aspect']})"])

    def _m_banner(self, frame):
        res = self.banner.detect(frame, draw=frame)
        if res.confirmed:
            DETLOG.log("banner", "aerothon_green", bbox=res.bbox,
                       px=res.centroid, gps=self._gps(), state="DIAG")
        tag = "CONFIRMED" if res.confirmed else ("CANDIDATE" if res.found else "--")
        if res.found:
            cv2.putText(frame, f"BANNER {tag}", (res.bbox[0], res.bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 220, 80), 2)
        return self._hud(frame, [
            "CORRIDOR ENTRY DETECTION (S2)",
            f"green banner: {tag}  rectangularity={res.score:.2f}",
            "HSV candidate -> area/AR/rectangularity/edge -> persistence"])

    def _m_lowering(self, frame):
        L, w = self.low, frame.shape[1]
        # virtual drone nudge: the BODY_OFFSET_NED corrections S5a would send
        # are rendered as a camera pan, so the centering loop visibly converges
        if L["phase"] == "ALIGN-DESCEND":
            px, py = L.get("pan", (0.0, 0.0))
            M = np.float32([[1, 0, -px], [0, 1, -py]])
            frame = cv2.warpAffine(frame, M, (w, frame.shape[0]))
        results = self.qr.detect(frame)
        target = next((r for r in results
                       if self.mission_code in (None, r.data)), None)
        st = self.center.update(frame.shape, target)
        L["centred"], L["gate"], L["off"] = \
            st["centred"], st["gate_open"], st["off_px"]
        vision.annotate_qrs(frame, results, mission_code=self.mission_code)

        h2, w2 = frame.shape[0] // 2, w // 2
        tol = int(w * config.VISION["center_tol_frac"])
        cv2.rectangle(frame, (w2 - tol, h2 - tol), (w2 + tol, h2 + tol),
                      (80, 220, 80) if st["centred"] else (40, 160, 240), 2)
        cv2.drawMarker(frame, (w2, h2), (245, 245, 245), cv2.MARKER_CROSS, 28, 1)

        if L["phase"] == "ALIGN-DESCEND":
            if st["visible"]:
                # nudge toward target (S5a 0.25 m increments, proportional)
                px, py = L.get("pan", (0.0, 0.0))
                L["pan"] = (px + 0.22 * st["off_px"][0],
                            py + 0.22 * st["off_px"][1])
                # centred -> descend; off-centre -> hold and nudge (S5a)
                if st["centred"]:
                    L["alt_m"] = max(config.MISSION["delivery_alt_m"],
                                     L["alt_m"] - 0.08)
            if L["alt_m"] <= config.MISSION["delivery_alt_m"] and L["gate"]:
                L["phase"] = "WINCH"
                self.winch.reset()
                self.winch.deliver()
        elif L["phase"] == "WINCH" and self.winch.state == "STOWED":
            L["phase"] = "COMPLETE"

        ws = self.winch.status()
        return self._hud(frame, [
            f"PAYLOAD DELIVERY DEMO (S5) | phase: {L['phase']}",
            f"alt {L['alt_m']:.1f} m -> 5.0 m | off ({st['off_px'][0]:+.0f},"
            f"{st['off_px'][1]:+.0f}) px | gate={'OPEN' if L['gate'] else 'closed'}",
            f"winch: {ws['state']} {ws['deployed_m']:.2f} m "
            f"({ws['deployed_pct']}%) hook_released={ws['hook_released']}"])

    # FSM dry-run stream: shows what the running mission FSM sees, with the
    # vision overlay matched to the ACTIVE state (Table 19 camera column) and
    # a telemetry strip (mode/armed/alt/hdg, LiDAR peaks, winch). Detection
    # here is display-only -- the FSM thread runs its own pipelines on the
    # same injected source; nothing shown is synthetic.
    _FSM_QR_STATES = ("S1", "S4", "S5a", "S5b", "S8")
    _FSM_BANNER_STATES = ("S2", "S3", "S6", "S7")

    def _m_fsm(self, frame):
        if not self.fsm:
            return self._hud(frame, [
                "FSM DRY RUN | press Start",
                "runs onboard/fsm.py on SIM MAVLink/LiDAR + REAL vision",
                "attach a camera / image / video first -- no input = "
                "camera failsafe (SX)"])
        snap = self.fsm.snapshot()
        sid = snap["state"]
        # state-matched vision overlay so the operator sees what the FSM sees
        try:
            if sid in self._FSM_QR_STATES:
                results = self.qr.detect(frame)
                vision.annotate_qrs(frame, results,
                                    mission_code=snap.get("mission_qr"))
            if sid in self._FSM_BANNER_STATES:
                res = self.banner.detect(frame, draw=frame)
                if res.found:
                    cv2.putText(frame, "BANNER " +
                                ("CONFIRMED" if res.confirmed else "CANDIDATE"),
                                (res.bbox[0], max(res.bbox[1] - 10, 14)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 220, 80), 2)
            if sid in ("S4", "S5a"):
                self.redzone.detect(frame, draw=frame)
        except Exception:
            pass

        # LiDAR strip (peak distance per sensor, ARM-FS trip highlighted)
        lid = ""
        try:
            scan = self.fsm.lidar.scan()
            parts = []
            for k in ("left", "front", "right"):
                p = scan[k]["peak_m"]
                parts.append(f"{k[0].upper()}:{p:.1f}m" if p is not None
                             else f"{k[0].upper()}:--")
            lid = ("LiDAR[" + snap["lidar"] + "] " + "  ".join(parts) +
                   ("  !TRIP" if scan.get("failsafe_trip") else ""))
        except Exception:
            lid = "LiDAR --"

        ws = snap["winch"]
        run = "RUNNING" if snap["running"] else \
            ("COMPLETE/STOPPED" if sid != "SX" else "FAILSAFE")
        last = snap["log"][-1]["msg"][:64] if snap["log"] else ""
        return self._hud(frame, [
            f"FSM {sid} {snap['phase']} | {run} | cam={snap['camera']}",
            f"{snap['mode']} {'ARMED' if snap['armed'] else 'disarmed'} | "
            f"alt {snap['alt']:.1f} m hdg {snap['heading']:.0f} | "
            f"QR:{snap.get('mission_qr') or '--'}",
            lid,
            f"winch {ws['state']} {ws['deployed_m']:.2f} m | {last}"])

    # --------------------------------------------------------- API helpers
    def _gps(self):
        if self.fsm and self.fsm.snapshot()["running"]:
            return self.fsm.mav.position()
        return None

    def lowering_cmd(self, action):
        if action == "start":
            self.low.update(phase="ALIGN-DESCEND", alt_m=10.0,
                            pan=(0.0, 0.0), t0=time.monotonic())
            self.center.reset()
        elif action == "reset":
            self.low.update(phase="STANDBY", alt_m=10.0, pan=(0.0, 0.0))
            self.winch.reset()
            self.center.reset()

    def fsm_cmd(self, action, speed=None):
        if action == "start":
            if self.fsm and self.fsm.snapshot()["running"]:
                return
            # honest provider: with no source attached the FSM receives None
            # and the camera-loss failsafe fires (as it would in flight) --
            # the NO INPUT card must never reach a detection pipeline
            self.fsm = MissionFSM(
                dry_run=True,
                time_scale=float(speed or 15.0),
                frame_provider=lambda _which:
                    None if self.src.kind == "none" else self.src.frame())
            self.fsm.start()
        elif action == "stop" and self.fsm:
            self.fsm.stop()

    def status(self):
        return {
            "source": self.src.kind, "src_info": self.src.info(),
            "mode": self.mode,
            "pyzbar": vision.PYZBAR_AVAILABLE,
            "mission_code": self.mission_code,
            "fence_configured": len(self.fence.data["official_fence"]) >= 3,
            "proc_scale": config.VISION.get("proc_scale", 1.0),
            "loop_fps": self._fps,
            "detlog_path": DETLOG.path,
            "clock": time.strftime("%H:%M:%S"),
        }


# ================================================================ HTTP layer
DIAG: Diagnostics = None  # type: ignore


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):                       # quiet
        pass

    # ----------------------------------------------------------- utilities
    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    # ------------------------------------------------------------------ GET
    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/":
            with open(os.path.join(ROOT, "webapp.html"), "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif p == "/stream":
            self._stream()
        elif p == "/api/specs":
            self._json(config.SPECS)
        elif p == "/api/status":
            self._json(DIAG.status())
        elif p == "/api/qr":
            self._json({"live": DIAG.last_qr, "log": DIAG.qr_log,
                        "mission_code": DIAG.mission_code})
        elif p == "/api/lowering":
            L = dict(DIAG.low)
            L["off"] = list(L["off"])
            L.pop("t0", None)
            self._json({"demo": L, "winch": DIAG.winch.status()})
        elif p == "/api/fsm":
            self._json(DIAG.fsm.snapshot() if DIAG.fsm else
                       {"state": "--", "phase": "NOT STARTED", "running": False,
                        "log": [], "winch": DIAG.winch.status()})
        elif p == "/api/settings":
            self._json(load_settings())
        elif p == "/api/snapshot":
            jpg = DIAG.snapshot_jpeg()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header(
                "Content-Disposition",
                f'attachment; filename="frame_{time.strftime("%H%M%S")}.jpg"')
            self.send_header("Content-Length", str(len(jpg)))
            self.end_headers()
            self.wfile.write(jpg)
        elif p == "/api/detlog":
            self._json({"path": DETLOG.path, "entries": DETLOG.tail(60)})
        elif p == "/api/detlog/download":
            try:
                with open(DETLOG.path, "rb") as f:
                    body = f.read()
            except Exception:
                body = b""
            self.send_response(200)
            self.send_header("Content-Type", "application/jsonl")
            self.send_header("Content-Disposition",
                             'attachment; filename="detections.jsonl"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif p == "/api/tune":
            self._json({k: config.VISION[k] for k in config.TUNABLE_KEYS})
        elif p == "/api/geofence":
            self._json(DIAG.fence.load())
        elif p == "/api/export":
            self._export()
        else:
            self._json({"error": "not found"}, 404)

    def _stream(self):
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        DIAG.clients += 1                      # wakes the loop from idle
        last = None
        try:
            while True:
                jpg = DIAG.jpeg()
                if jpg and jpg is not last:    # skip duplicate frames
                    last = jpg
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                     b"Content-Length: " +
                                     str(len(jpg)).encode() + b"\r\n\r\n")
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
                time.sleep(1.0 / config.VISION["target_fps"])
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            DIAG.clients = max(0, DIAG.clients - 1)

    def _export(self):
        buf = io.BytesIO()
        onboard = os.path.join(ROOT, "onboard")
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for root, _dirs, files in os.walk(onboard):
                for fn in files:
                    if fn.endswith((".pyc",)) or "__pycache__" in root:
                        continue
                    full = os.path.join(root, fn)
                    z.write(full, os.path.relpath(full, ROOT))
        body = buf.getvalue()
        self.send_response(200)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Disposition",
                         'attachment; filename="AT2026RC67_onboard.zip"')
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ----------------------------------------------------------------- POST
    def do_POST(self):
        p = self.path.split("?")[0]
        d = self._body()
        if p == "/api/mode":
            DIAG.set_mode(d.get("mode", "fpv"))
            self._json({"ok": True, "mode": DIAG.mode})
        elif p == "/api/source":
            t = d.get("type")
            name = d.get("name", t or "")
            ok = True
            err = None
            if t == "camera":
                ok = DIAG.src.set_camera(int(d.get("index", 0)))
                if not ok:
                    err = f"camera {d.get('index', 0)} failed to open"
            elif t == "image":
                try:
                    raw = base64.b64decode(d["data"].split(",")[-1])
                except Exception:
                    raw = b""
                img = cv2.imdecode(np.frombuffer(raw, np.uint8),
                                   cv2.IMREAD_COLOR)
                if img is None:
                    self._json({"ok": False, "error": "image decode failed",
                                "info": DIAG.src.info()}, 400)
                    return
                DIAG.src.set_image(img, name=name or "image")
            elif t == "video":
                try:
                    raw = base64.b64decode(d["data"].split(",")[-1])
                except Exception:
                    raw = b""
                suffix = os.path.splitext(name)[1] or ".mp4"
                tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
                tmp.write(raw)
                tmp.close()
                ok = DIAG.src.set_video(tmp.name, name=name or "video")
                if not ok:
                    err = "video failed to open (codec/container?)"
            elif t == "pause":
                DIAG.src.set_paused(bool(d.get("paused", True)))
            else:
                ok, err = False, f"unknown source type '{t}'"
            self._json({"ok": ok, "error": err, "info": DIAG.src.info()},
                       200 if ok else 400)
        elif p == "/api/qr/mission":
            if d.get("clear"):
                DIAG.mission_code = None
            elif d.get("auto"):
                DIAG.mission_code = None          # next confirmed QR captured
            elif "code" in d:
                DIAG.mission_code = d["code"]
            self._json({"ok": True, "mission_code": DIAG.mission_code})
        elif p == "/api/settings":
            self._json({"ok": True, "settings": save_settings(d)})
        elif p == "/api/tune":
            applied = {}
            for k, val in d.items():
                if k in config.TUNABLE_KEYS:
                    config.VISION[k] = tuple(val) if isinstance(val, list) \
                        else val
                    applied[k] = config.VISION[k]
            saved = config.save_tuning_overrides() if d.get("persist") else None
            self._json({"ok": True, "applied": applied, "saved_to": saved})
        elif p == "/api/tune/reset":
            try:
                os.remove(config.TUNING_FILE)
            except FileNotFoundError:
                pass
            import importlib
            importlib.reload(config)
            self._json({"ok": True,
                        "tune": {k: config.VISION[k]
                                 for k in config.TUNABLE_KEYS}})
        elif p == "/api/scale":
            s = float(d.get("scale", 1.0))
            config.VISION["proc_scale"] = min(max(s, 0.25), 1.0)
            self._json({"ok": True, "proc_scale": config.VISION["proc_scale"]})
        elif p == "/api/isolate":
            DIAG.iso.update({k: v for k, v in d.items() if k in DIAG.iso})
            self._json({"ok": True, "iso": DIAG.iso})
        elif p == "/api/lowering":
            DIAG.lowering_cmd(d.get("action", ""))
            self._json({"ok": True})
        elif p == "/api/fsm":
            DIAG.fsm_cmd(d.get("action", ""), d.get("speed"))
            self._json({"ok": True})
        elif p == "/api/geofence":
            DIAG.fence.save(d)
            self._json({"ok": True, "saved_to": DIAG.fence.path})
        else:
            self._json({"error": "not found"}, 404)


def main():
    global DIAG
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8742)
    ap.add_argument("--cam", type=int, default=None)
    a = ap.parse_args()
    s = load_settings()
    apply_settings(s)
    DIAG = Diagnostics(a.cam if a.cam is not None
                       else int(s.get("default_cam", 0)))
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), Handler)
    print(f"AT2026RC67 diagnostics bridge: http://localhost:{a.port}")
    print(f"  source: {DIAG.src.kind} | pyzbar: {vision.PYZBAR_AVAILABLE}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
