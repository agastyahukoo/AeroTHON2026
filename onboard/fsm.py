"""
fsm.py -- FM2 Finite State Machine, a 1:1 implementation of Design Report
Table 19 (Section 3.1) plus the expansion algorithms A-E and Section 4
failsafes.

  ID   PHASE                    CAMERA    LiDAR    MAVLink ON TRIGGER
  ---- ------------------------ --------- -------- ----------------------------
  S0a  PRE-ARM                  BOTH      ARM/TEST mode -> STABILIZE, ALTHOLD
  S0b  ARM                      --        DISARM   GUIDED_ENABLE, GUIDED_LIMITS,
                                                   mode -> GUIDED
  S1   TAKEOFF & SCAN           DOWN      DISARM   NAV_TAKEOFF + BODY_OFFSET_NED
  S2   FWD CORRIDOR ALIGNMENT   BOTH      DISARM   heading hold, Sideslip
                                                   Search Sweep (A)
  S3   FWD CORRIDOR NAV         FORWARD   ARM      Sideslip Obstacle Avoid (B)
  S4   DELIVERY ZONE SEARCH     DOWN      ARM-FS   SET_POSITION_TARGET_GLOBAL_INT
                                                   over fence-constrained grid (C)
  S5a  DELIVERY POSITIONING     DOWN      ARM-FS   BODY_OFFSET_NED nudges +
                                                   Vision-Assisted Positioning (D)
  S5b  PAYLOAD DELIVERY         DOWN      ARM-FS   hover, winch cycle
  S6   RTN CORRIDOR ALIGNMENT   --        ARM-FS   reciprocal heading (E)
  S7   RTN CORRIDOR NAV         FORWARD   ARM      Sideslip Obstacle Avoid (B)
  S8   LANDING & DATA LOG       DOWN      ARM-FS   RTL, delay 30 s, DISARM
  SX   FAILSAFE                 --        --       NAV_LAND

Run modes:
  live    -- real cameras / LiDAR / Pixhawk on the Pi 5
  dry_run -- SIM MAVLink + SIM LiDAR; vision runs the REAL detection
             pipelines on injected frames (camera / uploaded image / video).
             There are NO simulated detections anywhere -- if the target is
             not genuinely seen, the state times out and SX fires, exactly
             as in flight. The GCS "FSM dry run" diagnostic drives this.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import dataclass, field

import numpy as np

try:
    from . import config, vision, lidar as lidar_mod, geofence as gf_mod
    from .mavlink_io import Mavlink
    from .winch import Winch
    from .detlog import LOGGER as DETLOG
except ImportError:
    import config, vision, geofence as gf_mod
    import lidar as lidar_mod
    from mavlink_io import Mavlink
    from winch import Winch
    from detlog import LOGGER as DETLOG


# ---------------------------------------------------------------------------
@dataclass
class StateDef:
    sid: str
    phase: str
    camera: str          # BOTH / DOWN / FORWARD / --
    lidar_mode: str      # ARM / DISARM / ARM-FS / TEST


STATE_TABLE = [
    StateDef("S0a", "PRE-ARM",                "BOTH",    "TEST"),
    StateDef("S0b", "ARM",                    "--",      lidar_mod.DISARM),
    StateDef("S1",  "TAKEOFF & SCAN",         "DOWN",    lidar_mod.DISARM),
    StateDef("S2",  "FWD CORRIDOR ALIGNMENT", "BOTH",    lidar_mod.DISARM),
    StateDef("S3",  "FWD CORRIDOR NAV",       "FORWARD", lidar_mod.ARM),
    StateDef("S4",  "DELIVERY ZONE SEARCH",   "DOWN",    lidar_mod.ARM_FS),
    StateDef("S5a", "DELIVERY POSITIONING",   "DOWN",    lidar_mod.ARM_FS),
    StateDef("S5b", "PAYLOAD DELIVERY",       "DOWN",    lidar_mod.ARM_FS),
    StateDef("S6",  "RTN CORRIDOR ALIGNMENT", "FORWARD", lidar_mod.ARM_FS),
    StateDef("S7",  "RTN CORRIDOR NAV",       "FORWARD", lidar_mod.ARM),
    StateDef("S8",  "LANDING & DATA LOG",     "DOWN",    lidar_mod.ARM_FS),
    StateDef("SX",  "FAILSAFE",               "--",      "--"),
]
STATES = {s.sid: s for s in STATE_TABLE}


@dataclass
class MissionContext:
    mission_qr: str | None = None          # decoded delivery info (S1)
    corridor_start: tuple | None = None    # CORRIDOR_START_WPT (lat, lon)
    corridor_end: tuple | None = None      # CORRIDOR_END_WPT
    crdr_hdg: float | None = None          # corridor heading
    search_wpts: list = field(default_factory=list)
    search_idx: int = 0
    t_mission_start: float = 0.0
    log: list = field(default_factory=list)


class MissionFSM:
    def __init__(self, dry_run=True, frame_provider=None, time_scale=1.0,
                 on_event=None):
        """frame_provider(camera) -> BGR frame | None. In live mode the
        mission entry point wires Picamera2 capture here; in dry_run the
        GCS-selected source (camera / image / video) is injected. Detection
        is always real -- no synthetic results."""
        self.dry_run = dry_run
        self.time_scale = max(time_scale, 0.1)
        self.frame_provider = frame_provider or self._no_provider
        self._cam_misses = 0
        self.on_event = on_event or (lambda e: None)

        self.mav = Mavlink(force_sim=dry_run)
        self.lidar = lidar_mod.LidarArray(simulate=dry_run)
        self.winch = Winch(simulate=dry_run, time_scale=time_scale)
        self.fence = gf_mod.GeofenceManager()
        self.qr = vision.QRPipeline()
        self.banner = vision.BannerPipeline()
        self.redzone = vision.RedZonePipeline()
        self.center = vision.TargetCentering()

        self.ctx = MissionContext()
        self.state: StateDef = STATES["S0a"]
        self._running = threading.Event()
        self._thread = None

    # ------------------------------------------------------------ plumbing
    @staticmethod
    def _no_provider(_cam):
        return None                                    # no provider = no frames

    def _log(self, msg, level="INFO"):
        e = {"t": round(time.monotonic() - self.ctx.t_mission_start, 2),
             "state": self.state.sid, "phase": self.state.phase,
             "level": level, "msg": msg}
        self.ctx.log.append(e)
        self.on_event(e)

    def _enter(self, sid):
        self.state = STATES[sid]
        if self.state.lidar_mode in (lidar_mod.ARM, lidar_mod.DISARM,
                                     lidar_mod.ARM_FS):
            self.lidar.set_mode(self.state.lidar_mode)
        self._log(f"ENTER {sid} {self.state.phase} | cam={self.state.camera} "
                  f"lidar={self.state.lidar_mode}")

    def _sleep(self, s):
        t_end = time.monotonic() + s / self.time_scale
        while time.monotonic() < t_end and self._running.is_set():
            time.sleep(0.05)

    def _frame(self, which):
        f = self.frame_provider(which)
        if f is None:
            self._cam_misses += 1
            return np.zeros((config.VISION["frame_h"] // 2,
                             config.VISION["frame_w"] // 2, 3), np.uint8)
        self._cam_misses = 0
        return f

    # ------------------------------------------------- continuous guarding
    def _guard(self) -> bool:
        """Per-tick failsafe sweep (Section 4). Returns False => go SX."""
        if not self._running.is_set():
            return False
        if not self.mav.heartbeat_ok():
            self._log("FC heartbeat lost -> FAILSAFE", "CRIT")
            return False
        if self._cam_misses > 45:                      # ~3 s of lost frames
            self._log("Camera feed lost -> FAILSAFE", "CRIT")
            return False
        lat, lon, alt = self.mav.position()
        chk = self.fence.check(lat, lon, alt)
        if chk["configured"] and not chk["inside_obc"]:
            self._log("OBC geofence exceeded -> FAILSAFE", "CRIT")
            return False
        if self.mav.fence_breached():
            self._log("Pixhawk FENCE_STATUS breach -> FAILSAFE", "CRIT")
            return False
        scan = self.lidar.scan()
        if scan["failsafe_trip"]:
            self._log("LiDAR ARM-FS proximity trip -> FAILSAFE", "CRIT")
            return False
        if (time.monotonic() - self.ctx.t_mission_start) * self.time_scale \
                > config.MISSION["time_limit_s"]:
            self._log("Mission time limit -> FAILSAFE", "CRIT")
            return False
        return True

    # ============================================================== states
    def _s0a_pre_arm(self):
        self._enter("S0a")
        self.mav.set_mode("STABILIZE")
        self._sleep(0.5)
        self.mav.set_mode("ALT_HOLD")
        ok_cam = self.frame_provider("down") is not None and \
            self.frame_provider("forward") is not None
        ok_lidar = all(k in self.lidar.scan() for k in ("left", "front", "right"))
        self._log(f"Checks: camera={'OK' if ok_cam else 'FAIL'} "
                  f"lidar={'OK' if ok_lidar else 'FAIL'} "
                  f"mavlink={'OK' if self.mav.heartbeat_ok() else 'FAIL'}")
        return ok_cam and ok_lidar and self.mav.heartbeat_ok()

    def _s0b_arm(self):
        self._enter("S0b")
        self.mav.guided_enable()
        lim = self.mav.guided_limits()
        self._log(f"MAV_CMD_DO_GUIDED_LIMITS set: alt<={lim['max_alt_m']}m "
                  f"horiz<={lim['max_horiz_m']}m t<={lim['timeout_s']}s")
        self.mav.set_mode("GUIDED")
        self.mav.arm(True)
        self._log("Armed. Holding 10 s to permit disarm if desired.")
        self._sleep(10.0)
        return True

    def _s1_takeoff_scan(self):
        self._enter("S1")
        self.mav.takeoff(config.MISSION["takeoff_alt_m"])
        self._sleep(3.0)
        self.mav.body_offset_ned(fwd=config.MISSION["scan_forward_m"])
        self._log("At 5 m, moved fwd ~1 m. Scanning start QR (down camera).")
        t0 = time.monotonic()
        while self._running.is_set() and time.monotonic() - t0 < 60 / self.time_scale:
            frame = self._frame("down")
            for r in self.qr.detect(frame):
                if r.confirmed:
                    self.ctx.mission_qr = r.data
                    DETLOG.log("qr", r.data, px=r.centroid,
                               gps=self.mav.position(), decoder=r.decoder,
                               state=self.state.sid, extra={"role": "start"})
                    self._log(f"Start QR decoded [{r.decoder}]: '{r.data}' "
                              "-> target stored in memory")
                    return True
            time.sleep(0.03)
        return False

    def _s2_fwd_alignment(self):
        self._enter("S2")
        # Sideslip Search Sweep (A): roll-only perturbations, LiDAR DISARMED
        step, sign, n = config.MISSION["sideslip_step_m"], 1, 0
        t0 = time.monotonic()
        while self._running.is_set() and time.monotonic() - t0 < 90 / self.time_scale:
            res = self.banner.detect(self._frame("forward"))
            if res.confirmed:
                DETLOG.log("banner", "aerothon_green", bbox=res.bbox,
                           px=res.centroid, gps=self.mav.position(),
                           state=self.state.sid)
                lat, lon, _ = self.mav.position()
                self.ctx.corridor_start = (lat, lon)
                self.ctx.crdr_hdg = self.mav.state["heading"]
                self._log("Green banner centred. CORRIDOR_START_WPT + CRDR_HDG"
                          f" saved ({lat:.6f},{lon:.6f} hdg={self.ctx.crdr_hdg:.0f})")
                self.mav.body_offset_ned(
                    down=config.MISSION["takeoff_alt_m"]
                    - config.MISSION["corridor_alt_m"])
                self._log("Descended to 3 m corridor altitude.")
                return True
            self.mav.body_offset_ned(right=sign * step * (n // 2 + 1))
            sign, n = -sign, n + 1
            self._sleep(0.6)
        return False

    def _corridor_nav(self, sid, exit_check):
        """Shared S3/S7 body: Sideslip Obstacle Avoidance (B)."""
        self._enter(sid)
        trim_lim = config.MISSION["heading_trim_deg"]
        while self._running.is_set():
            if not self._guard():
                return False
            scan = self.lidar.scan()
            frame = self._frame("forward")
            # vision lateral offset estimate fused with LiDAR via Weighted-LPF
            res = self.banner.detect(frame)
            v_off = 0.0
            if res.found:
                v_off = (res.centroid[0] - frame.shape[1] / 2) / \
                    (frame.shape[1] / 2) * (config.MISSION["corridor_width_m"] / 2)
            fused = self.lidar.fuse_lateral(v_off, scan)

            if scan["forward_obstacle"]:
                side = -1 if scan["left"]["wall_m"] > scan["right"]["wall_m"] else 1
                self._log(f"Forward LiDAR obstacle @{scan['front']['peak_m']:.2f}m"
                          f" -> sideslip {'L' if side < 0 else 'R'}")
                self.mav.body_offset_ned(
                    right=side * config.MISSION["sideslip_avoid_step_m"])
            else:
                self.mav.body_offset_ned(fwd=config.MISSION["sideslip_step_m"])
                # heading trim along corridor walls, bounded +/-15deg (B)
                trim = max(-trim_lim, min(trim_lim, -fused * 12.0))
                if abs(trim) > 2.0:
                    self.mav.condition_yaw(trim * 0.25)
            if exit_check():
                return True
            self._sleep(0.4)
        return False

    def _dist_from(self, wpt):
        if wpt is None:
            return 0.0
        lat, lon, _ = self.mav.position()
        dn = (lat - wpt[0]) * 111_111.0
        de = (lon - wpt[1]) * 111_111.0 * math.cos(math.radians(lat))
        return math.hypot(dn, de)

    def _s3(self):
        ok = self._corridor_nav(
            "S3", lambda: self._dist_from(self.ctx.corridor_start)
            > config.MISSION["corridor_min_len_m"])
        if ok:
            lat, lon, _ = self.mav.position()
            self.ctx.corridor_end = (lat, lon)
            self._log(f"CORRIDOR_END_WPT logged ({lat:.6f},{lon:.6f})")
        return ok

    def _s4_delivery_search(self):
        self._enter("S4")
        self.mav.body_offset_ned(
            down=-(config.MISSION["delivery_search_alt_m"]
                   - config.MISSION["corridor_alt_m"]))
        self._log("Ascended to 10 m. Running geofence-constrained grid (C).")
        self.ctx.search_wpts = self.fence.delivery_search_pattern() or \
            [list(self.mav.position()[:2])]
        for i, (lat, lon) in enumerate(self.ctx.search_wpts):
            if not self._guard():
                return False
            self.ctx.search_idx = i
            self.mav.goto_global(lat, lon,
                                 config.MISSION["delivery_search_alt_m"])
            for _ in range(12):
                frame = self._frame("down")
                rz = self.redzone.detect(frame)
                if rz.roi_intruded:
                    if rz.zones:
                        DETLOG.log("redzone", "roi_intrusion",
                                   bbox=rz.zones[0]["bbox"],
                                   px=rz.zones[0]["centroid"],
                                   gps=self.mav.position(),
                                   state=self.state.sid)
                    self._log(f"Red zone in ROI -> yaw "
                              f"{'L' if rz.avoid_yaw < 0 else 'R'} (yaw-only)")
                    self.mav.condition_yaw(15 * rz.avoid_yaw)
                for r in self.qr.detect(frame):
                    if r.data == self.ctx.mission_qr and r.confirmed:
                        DETLOG.log("qr", r.data, px=r.centroid,
                                   gps=self.mav.position(), decoder=r.decoder,
                                   state=self.state.sid,
                                   extra={"role": "delivery_match"})
                        self._log(f"Matching delivery QR found: '{r.data}'")
                        return True
                self._sleep(0.15)
        return False

    def _s5a_positioning(self):
        self._enter("S5a")
        self.center.reset()
        t0 = time.monotonic()
        while self._running.is_set() and time.monotonic() - t0 < 45 / self.time_scale:
            if not self._guard():
                return False
            frame = self._frame("down")
            roll, pitch = self.mav.attitude()
            frame, _ = self.center.stabilise(frame, roll, pitch)   # DIS
            match = next((r for r in self.qr.detect(frame)
                          if r.data == self.ctx.mission_qr), None)
            st = self.center.update(frame.shape, match)
            if st["gate_open"]:
                self._log("QR centred within tolerance (persistent) -> "
                          "descend to 5 m.")
                self.mav.body_offset_ned(
                    down=config.MISSION["delivery_search_alt_m"]
                    - config.MISSION["delivery_alt_m"])
                return True
            if st["visible"]:
                nx, ny = st["off_norm"]
                self.mav.body_offset_ned(
                    fwd=-ny * config.MISSION["nudge_step_m"],
                    right=nx * config.MISSION["nudge_step_m"])
            self._sleep(0.3)
        return False

    def _s5b_delivery(self):
        self._enter("S5b")
        self._log("Hover hold 5 s to stabilise, then full pulley spool-out.")
        self._sleep(5.0)
        self.winch.deliver(blocking=True)
        ok = self.winch.state == "STOWED" and self.winch.hook_released
        self._log("Payload lowered, gravity hook released, line retracted, "
                  "pretensioned." if ok else "Winch cycle incomplete.", 
                  "INFO" if ok else "WARN")
        return ok

    def _s6_rtn_alignment(self):
        self._enter("S6")
        # Corridor Re-Entry (E): ascend to return altitude, fly back to
        # CORRIDOR_END_WPT and take the reciprocal heading
        self.mav.body_offset_ned(
            down=-(config.MISSION["delivery_search_alt_m"]
                   - config.MISSION["delivery_alt_m"]))
        if self.ctx.corridor_end:
            self.mav.goto_global(*self.ctx.corridor_end,
                                 config.MISSION["delivery_search_alt_m"])
        recip = ((self.ctx.crdr_hdg or 0) + 180.0) % 360.0
        self.mav.set_heading(recip)
        self._log(f"CORRIDOR_END_WPT reached, reciprocal heading {recip:.0f}deg.")
        # Corridor Entry Detection -- Return Lap (rulebook 4.2.4): re-detect
        # the green AeroTHON banner with the forward camera and align before
        # corridor navigation. WPT/heading remain the fallback if the banner
        # is not re-acquired within the window (logged either way).
        self.banner.reset()
        step, sign, n = config.MISSION["sideslip_step_m"], 1, 0
        t0 = time.monotonic()
        while self._running.is_set() and \
                time.monotonic() - t0 < 30 / self.time_scale:
            if not self._guard():
                return False
            res = self.banner.detect(self._frame("forward"))
            if res.confirmed:
                DETLOG.log("banner", "aerothon_green_return", bbox=res.bbox,
                           px=res.centroid, gps=self.mav.position(),
                           state=self.state.sid, extra={"lap": "return"})
                self._log("Return-lap banner re-acquired -> aligned with "
                          "corridor entrance.")
                break
            # bounded sideslip sweep while searching (algorithm A pattern)
            self.mav.body_offset_ned(right=sign * step * (n // 2 + 1))
            sign, n = -sign, n + 1
            self._sleep(0.5)
        else:
            self._log("Return banner not re-acquired in window -> "
                      "falling back to CORRIDOR_END_WPT + reciprocal heading.",
                      "WARN")
        # descend to corridor navigation altitude for the return lap
        self.mav.body_offset_ned(
            down=config.MISSION["delivery_search_alt_m"]
            - config.MISSION["corridor_alt_m"])
        self._log("Descended to 3 m corridor altitude (return lap).")
        return True

    def _s7(self):
        return self._corridor_nav(
            "S7", lambda: self._dist_from(self.ctx.corridor_end)
            > config.MISSION["corridor_min_len_m"] or
            self._dist_from(self.ctx.corridor_start) < 2.0)

    def _s8_landing(self):
        self._enter("S8")
        self.mav.rtl()
        self._log("RTL commanded (vision-assisted landing active).")
        self._sleep(config.MISSION["rtl_disarm_delay_s"])
        self.mav.arm(False)
        self._log("Disarmed. Writing mission log.")
        self._write_log()
        return True

    def _sx_failsafe(self):
        self._enter("SX")
        self.winch.abort()
        self.mav.land()
        self._log("MAV_CMD_NAV_LAND issued (SX).", "CRIT")
        self._write_log()

    def _write_log(self):
        path = config.LOGGING["mission_log"]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.ctx.log, f, indent=2)

    # ============================================================ lifecycle
    def start(self):
        if self._thread and self._thread.is_alive():
            return False
        self._running.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._running.clear()

    def _run(self):
        self.ctx = MissionContext(t_mission_start=time.monotonic())
        seq = [self._s0a_pre_arm, self._s0b_arm, self._s1_takeoff_scan,
               self._s2_fwd_alignment, self._s3, self._s4_delivery_search,
               self._s5a_positioning, self._s5b_delivery,
               self._s6_rtn_alignment, self._s7, self._s8_landing]
        for step in seq:
            if not self._running.is_set() or not step():
                if self._running.is_set():
                    self._sx_failsafe()
                else:
                    self._log("Mission stopped by operator.", "WARN")
                return
        self._log("MISSION COMPLETE.", "INFO")

    def snapshot(self) -> dict:
        lat, lon, alt = self.mav.position()
        return {"state": self.state.sid, "phase": self.state.phase,
                "camera": self.state.camera, "lidar": self.lidar.mode,
                "mode": self.mav.state["mode"], "armed": self.mav.state["armed"],
                "lat": lat, "lon": lon, "alt": round(alt, 2),
                "heading": round(self.mav.state["heading"], 1),
                "mission_qr": self.ctx.mission_qr,
                "winch": self.winch.status(),
                "running": self._running.is_set() and self._thread is not None
                and self._thread.is_alive(),
                "log": self.ctx.log[-200:]}
