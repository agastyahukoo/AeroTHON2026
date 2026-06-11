"""
mavlink_io.py -- Pixhawk 2.4.8 link per Design Report 1.9 / 3.1 / Safety A-B.

Every MAVLink primitive named in FSM Table 19 is implemented here:

  MODE CHANGE -> STABILIZE / ALTHOLD / GUIDED        (S0)
  MAV_CMD_NAV_GUIDED_ENABLE, MAV_CMD_DO_GUIDED_LIMITS (S0 anti-flyaway)
  MAV_CMD_NAV_TAKEOFF                                 (S1)
  MAV_FRAME_BODY_OFFSET_NED position targets          (S2/S3/S5/S6/S7 sideslip)
  SET_POSITION_TARGET_GLOBAL_INT                      (S4 geofenced grid search)
  Conditional yaw (red-zone avoidance, yaw-only)      (3.2)
  MAV_CMD_NAV_RETURN_TO_LAUNCH, MAV_CMD_NAV_LAND      (S8/SX)
  MAV_CMD_COMPONENT_ARM_DISARM                        (S0/S8)
  FENCE_STATUS read (breach_status)                   (Safety: Geofence Config)
  Heartbeat watchdog (loss of OBC link -> LAND)       (1.1 custom failsafes)
  ATTITUDE stream (roll/pitch for DIS warpAffine)     (Innovation 3)

Connection: TELEM2 <-> Pi GPIO14/15 UART, no 5V line (report 1.9).
If pymavlink is absent or the port cannot be opened, a kinematic SIM backend
keeps the identical interface so the FSM and GCS diagnostics run anywhere.
"""

from __future__ import annotations

import math
import threading
import time

try:
    from . import config
except ImportError:
    import config

try:
    from pymavlink import mavutil
    PYMAVLINK_AVAILABLE = True
except Exception:
    mavutil = None
    PYMAVLINK_AVAILABLE = False

COPTER_MODES = {"STABILIZE": 0, "ALT_HOLD": 2, "GUIDED": 4,
                "LOITER": 5, "RTL": 6, "LAND": 9}


class Mavlink:
    """Single facade; .sim is True when running the kinematic backend."""

    def __init__(self, connection: str | None = None, force_sim: bool = False):
        self.sim = force_sim or not PYMAVLINK_AVAILABLE
        self._lock = threading.Lock()
        self.state = {                                  # mirrored telemetry
            "mode": "STABILIZE", "armed": False,
            "lat": 19.1076, "lon": 72.8370, "rel_alt": 0.0,   # MPSTME campus
            "heading": 0.0, "roll": 0.0, "pitch": 0.0,
            "vbatt": 16.6, "fence_breach": 0,
            "last_heartbeat": time.monotonic(),
        }
        self._home = None
        if not self.sim:
            try:
                self.master = mavutil.mavlink_connection(
                    connection or config.MAVLINK["connection"],
                    baud=config.MAVLINK["baud"],
                    source_system=config.MAVLINK["system_id"])
                self.master.wait_heartbeat(timeout=5)
                threading.Thread(target=self._rx_loop, daemon=True).start()
            except Exception:
                self.sim = True
        if self.sim:
            self.master = None

    # ------------------------------------------------------------ RX side
    def _rx_loop(self):                                # pragma: no cover
        while True:
            msg = self.master.recv_match(blocking=True, timeout=1)
            if msg is None:
                continue
            t = msg.get_type()
            with self._lock:
                if t == "HEARTBEAT":
                    self.state["last_heartbeat"] = time.monotonic()
                    self.state["armed"] = bool(
                        msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                elif t == "GLOBAL_POSITION_INT":
                    self.state.update(lat=msg.lat / 1e7, lon=msg.lon / 1e7,
                                      rel_alt=msg.relative_alt / 1000.0,
                                      heading=msg.hdg / 100.0)
                elif t == "ATTITUDE":
                    self.state.update(roll=math.degrees(msg.roll),
                                      pitch=math.degrees(msg.pitch))
                elif t == "SYS_STATUS":
                    self.state["vbatt"] = msg.voltage_battery / 1000.0
                elif t == "FENCE_STATUS":
                    self.state["fence_breach"] = msg.breach_status

    def heartbeat_ok(self) -> bool:
        return (time.monotonic() - self.state["last_heartbeat"]
                < config.MAVLINK["heartbeat_timeout_s"]) or self.sim

    # ----------------------------------------------------------- commands
    def _cmd_long(self, cmd, *params):                 # pragma: no cover
        p = list(params) + [0] * (7 - len(params))
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            cmd, 0, *p)

    def set_mode(self, mode: str):
        with self._lock:
            self.state["mode"] = mode
        if not self.sim:                               # pragma: no cover
            self.master.set_mode(COPTER_MODES[mode])

    def guided_enable(self):
        if not self.sim:                               # pragma: no cover
            self._cmd_long(mavutil.mavlink.MAV_CMD_NAV_GUIDED_ENABLE, 1)

    def guided_limits(self):
        """MAV_CMD_DO_GUIDED_LIMITS: time / alt floor & ceiling / horizontal
        move limit. Makes an OBC-induced flyaway practically improbable."""
        g = config.MAVLINK["guided_limits"]
        if not self.sim:                               # pragma: no cover
            self._cmd_long(mavutil.mavlink.MAV_CMD_DO_GUIDED_LIMITS,
                           g["timeout_s"], g["min_alt_m"], g["max_alt_m"],
                           g["max_horiz_m"])
        return g

    def arm(self, arm=True):
        with self._lock:
            self.state["armed"] = arm
            if arm and self._home is None:
                self._home = (self.state["lat"], self.state["lon"])
        if not self.sim:                               # pragma: no cover
            self._cmd_long(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                           1 if arm else 0)

    def takeoff(self, alt_m: float):
        if self.sim:
            self._sim_alt_to(alt_m)
        else:                                          # pragma: no cover
            self._cmd_long(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0,
                           0, 0, alt_m)

    def body_offset_ned(self, fwd=0.0, right=0.0, down=0.0):
        """SET_POSITION_TARGET_LOCAL_NED in MAV_FRAME_BODY_OFFSET_NED --
        the primitive behind the Sideslip Search / Avoidance algorithms."""
        if self.sim:
            with self._lock:
                hdg = math.radians(self.state["heading"])
                dn = fwd * math.cos(hdg) - right * math.sin(hdg)
                de = fwd * math.sin(hdg) + right * math.cos(hdg)
                self.state["lat"] += dn / 111_111.0
                self.state["lon"] += de / (111_111.0 *
                                           math.cos(math.radians(self.state["lat"])))
                self.state["rel_alt"] = max(0.0, self.state["rel_alt"] - down)
            return
        m = self.master.mav                            # pragma: no cover
        m.set_position_target_local_ned_send(
            0, self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
            0b0000111111111000, fwd, right, down,
            0, 0, 0, 0, 0, 0, 0, 0)

    def goto_global(self, lat, lon, alt_m):
        """SET_POSITION_TARGET_GLOBAL_INT -- geofence-constrained search (C)."""
        if self.sim:
            with self._lock:
                self.state.update(lat=lat, lon=lon)
            self._sim_alt_to(alt_m)
            return
        m = self.master.mav                            # pragma: no cover
        m.set_position_target_global_int_send(
            0, self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            0b0000111111111000, int(lat * 1e7), int(lon * 1e7), alt_m,
            0, 0, 0, 0, 0, 0, 0, 0)

    def condition_yaw(self, deg_rel: float):
        """Yaw-only red-zone avoidance (no induced roll)."""
        if self.sim:
            with self._lock:
                self.state["heading"] = (self.state["heading"] + deg_rel) % 360
            return
        self._cmd_long(mavutil.mavlink.MAV_CMD_CONDITION_YAW,  # pragma: no cover
                       abs(deg_rel), 10, 1 if deg_rel >= 0 else -1, 1)

    def set_heading(self, deg_abs: float):
        if self.sim:
            with self._lock:
                self.state["heading"] = deg_abs % 360
        else:                                          # pragma: no cover
            self._cmd_long(mavutil.mavlink.MAV_CMD_CONDITION_YAW,
                           deg_abs % 360, 10, 0, 0)

    def rtl(self):
        self.set_mode("RTL")
        if self.sim and self._home:
            with self._lock:
                self.state["lat"], self.state["lon"] = self._home
            self._sim_alt_to(0.0)
        elif not self.sim:                             # pragma: no cover
            self._cmd_long(mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH)

    def land(self):
        self.set_mode("LAND")
        if self.sim:
            self._sim_alt_to(0.0)
        else:                                          # pragma: no cover
            self._cmd_long(mavutil.mavlink.MAV_CMD_NAV_LAND)

    # ------------------------------------------------------------- helpers
    def _sim_alt_to(self, alt):
        with self._lock:
            self.state["rel_alt"] = float(alt)

    def fence_breached(self) -> bool:
        return self.state["fence_breach"] == 1

    def attitude(self):
        return self.state["roll"], self.state["pitch"]

    def position(self):
        return self.state["lat"], self.state["lon"], self.state["rel_alt"]
