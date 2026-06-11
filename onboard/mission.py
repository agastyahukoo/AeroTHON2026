#!/usr/bin/env python3
"""
mission.py -- FM2 entry point. This file runs on the Raspberry Pi 5 at the
flight line.

Usage on the drone (live):
    python3 mission.py                    # full autonomous FM2

Demo / bench functions (per team request these live INSIDE the onboard code
so the exact flight modules are what gets exercised):
    python3 mission.py --dry-run [--speed 10]   # full FSM on simulators
    python3 mission.py --demo qr                # live QR pipeline on camera 0
    python3 mission.py --demo banner            # banner pipeline on camera 0
    python3 mission.py --demo redzone           # red-zone pipeline on camera 0
    python3 mission.py --demo lidar             # LiDAR scan printout (sim/hw)
    python3 mission.py --demo winch             # accelerated winch cycle
    python3 mission.py --demo geofence          # fence check + search grid
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

import cv2
import numpy as np

import config

cv2.setUseOptimized(True)                            # Pi 5: NEON fast paths
cv2.setNumThreads(config.VISION.get("cv_threads", 3))
import vision
import lidar as lidar_mod
import geofence as gf_mod
from fsm import MissionFSM
from winch import Winch


# ---------------------------------------------------------------- cameras
class CameraRig:
    """Live capture: forward = piloting IMX708, down = spotting IMX708.
    On the Pi 5 the CSI cameras run through Picamera2/libcamera (hardware
    ISP, near-zero CPU); cv2.VideoCapture covers bench USB webcams."""

    def __init__(self):
        v = config.VISION
        self._picams, self._caps = {}, {}
        try:
            from picamera2 import Picamera2
            for name, idx in (("forward", v["forward_cam_index"]),
                              ("down", v["down_cam_index"])):
                try:
                    cam = Picamera2(idx)
                    cam.configure(cam.create_video_configuration(
                        main={"size": (v["frame_w"], v["frame_h"]),
                              "format": "RGB888"}))
                    cam.start()
                    self._picams[name] = cam
                except Exception:
                    pass
        except Exception:
            pass
        for name, idx in (("forward", v["forward_cam_index"]),
                          ("down", v["down_cam_index"])):
            if name in self._picams:
                continue
            cap = cv2.VideoCapture(idx)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, v["frame_w"])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, v["frame_h"])
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._caps[name] = cap if cap.isOpened() else None

        # Pi 5 budget: one grabber thread per camera keeps only the LATEST
        # frame, so the FSM never blocks on capture or processes stale buffers
        self._latest, self._llock = {}, threading.Lock()
        for name in set(self._picams) | {k for k, v in self._caps.items() if v}:
            threading.Thread(target=self._grab, args=(name,), daemon=True).start()

    def _grab(self, name):
        while True:
            frame = None
            cam = self._picams.get(name)
            if cam is not None:
                try:
                    frame = cam.capture_array()        # RGB888 == BGR layout
                except Exception:
                    frame = None
            else:
                cap = self._caps.get(name)
                if cap is not None:
                    ok, f = cap.read()
                    frame = f if ok else None
            if frame is None:
                time.sleep(0.05)
                continue
            with self._llock:
                self._latest[name] = frame

    def __call__(self, which):
        """Latest frame for 'forward'/'down', or None -- camera loss is real
        and surfaces as an FSM failsafe; no placeholder frames are produced."""
        with self._llock:
            f = self._latest.get(which)
            if f is None and self._latest:             # single-camera bench rig
                f = next(iter(self._latest.values()))
        return None if f is None else f.copy()


# ------------------------------------------------------------------ demos
def demo_qr():
    rig, qr = CameraRig(), vision.QRPipeline()
    print("[demo] QR pipeline: OpenCV<->pyzbar toggling, persistence gate. "
          "q to quit.")
    while True:
        frame = rig("down")
        results = qr.detect(frame)
        vision.annotate_qrs(frame, results)
        for r in results:
            if r.confirmed:
                print(f"  CONFIRMED [{r.decoder}] {r.data}")
        cv2.imshow("QR", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def demo_banner():
    rig, bp = CameraRig(), vision.BannerPipeline()
    print("[demo] Green-banner hybrid pipeline. q to quit.")
    while True:
        frame = rig("forward")
        res = bp.detect(frame, draw=frame)
        cv2.putText(frame, f"banner={'CONFIRMED' if res.confirmed else ('SEEN' if res.found else '--')}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 220, 80), 2)
        cv2.imshow("Banner", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def demo_redzone():
    rig, rz = CameraRig(), vision.RedZonePipeline()
    print("[demo] Red restricted-zone pipeline. q to quit.")
    while True:
        frame = rig("down")
        res = rz.detect(frame, draw=frame)
        cv2.putText(frame, f"ROI intruded={res.roi_intruded} yaw={res.avoid_yaw:+d}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 230), 2)
        cv2.imshow("RedZone", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def demo_lidar():
    arr = lidar_mod.LidarArray()
    arr.set_mode(lidar_mod.ARM)
    print(f"[demo] LiDAR array ({'SIM' if arr.simulate else 'HW'}). Ctrl-C to quit.")
    try:
        while True:
            s = arr.scan()
            print(f"  L={s['left']['wall_m']:.2f}m  F-peak="
                  f"{s['front']['peak_m']}  R={s['right']['wall_m']:.2f}m  "
                  f"lat-innov={s['lateral_innovation_m']:+.2f}m  "
                  f"fwd-obst={s['forward_obstacle']}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass


def demo_winch():
    w = Winch(time_scale=20.0)
    print("[demo] Winch cycle at 20x (real cycle: 106 s out + settles).")
    w.deliver()
    while w.state != "STOWED":
        st = w.status()
        print(f"  {st['state']:<10} {st['deployed_m']:.2f} m "
              f"({st['deployed_pct']}%)  hook_released={st['hook_released']}")
        time.sleep(0.4)
    print("  STOWED. Done.")


def demo_geofence():
    g = gf_mod.GeofenceManager()
    print("[demo] Geofence:", "configured" if g.data["official_fence"] else
          "NOT configured (use GCS mission-planning page)")
    if g.data["official_fence"]:
        lat, lon = g.data["official_fence"][0]
        print("  check@corner:", g.check(lat, lon, 5))
        print(f"  S4 search waypoints: {len(g.delivery_search_pattern())}")


def run_mission(dry_run: bool, speed: float):
    provider = None if dry_run else CameraRig()
    fsm = MissionFSM(dry_run=dry_run, frame_provider=provider,
                     time_scale=speed,
                     on_event=lambda e: print(
                         f"[{e['t']:>7.1f}s][{e['state']:>3}] "
                         f"{e['level']:<4} {e['msg']}"))
    fsm.start()
    try:
        while fsm.snapshot()["running"]:
            time.sleep(0.5)
    except KeyboardInterrupt:
        fsm.stop()
    print("Mission thread ended.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="AT2026RC67 FM2 mission")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--demo", choices=["qr", "banner", "redzone", "lidar",
                                       "winch", "geofence"])
    a = ap.parse_args()
    if a.demo:
        {"qr": demo_qr, "banner": demo_banner, "redzone": demo_redzone,
         "lidar": demo_lidar, "winch": demo_winch,
         "geofence": demo_geofence}[a.demo]()
        sys.exit(0)
    run_mission(dry_run=a.dry_run, speed=a.speed)
