"""
lidar.py -- Triple VL53L7CX ToF array per Design Report 1.9 / 3.2.

3 solid-state 8x8 multizone LiDARs (left / front / right) give ~270deg frontal
coverage. Two algorithms exactly as documented:

  1. Peak threshold detection -- closest cell accepted ONLY if corroborated by
     an adjacent cell (denoises outdoor sunlight glint false-positives).
     Primary algorithm in ARM-FS mode.
  2. Wall gradient detection -- a plane/slope is fit over the 8x8 mesh, giving
     inherently tilt-compensated wall distance without an external IMU or a
     MAVLink roll-polling loop. Used by the Sideslip Obstacle Avoidance loop.

Modes (FSM Table 19 column "LiDAR"):
  DISARM  -- ignored (S1/S2 sideslip search: false trigger would void points)
  ARM     -- obstacle data feeds sideslip avoidance (S3/S7 corridor)
  ARM-FS  -- any proximity below failsafe threshold => anomaly => RTL/LAND
             (EGPWS-style hard protection, S4..S8)

Hardware access uses the ST ULD via vl53l7cx python bindings when present;
otherwise a deterministic simulator stands in so the full stack (FSM, fusion,
GCS diagnostics) runs identically on a desktop.
"""

from __future__ import annotations

import math
import time

import numpy as np

try:
    from . import config
except ImportError:
    import config

try:                                                  # pragma: no cover
    import vl53l7cx  # ST ULD binding on the Pi 5
    HW_AVAILABLE = True
except Exception:
    vl53l7cx = None
    HW_AVAILABLE = False

DISARM, ARM, ARM_FS = "DISARM", "ARM", "ARM-FS"


class _SimSensor:
    """Deterministic corridor simulator: two walls 3.5 m apart, drifting pose,
    an occasional obstacle on the front sensor. Sufficient for dry runs."""

    def __init__(self, name):
        self.name = name
        self._t0 = time.monotonic()

    def ranging_data(self):
        g = config.LIDAR["grid"]
        t = time.monotonic() - self._t0
        drift = 0.45 * math.sin(t * 0.25)                       # lateral drift
        half = config.MISSION["corridor_width_m"] / 2.0
        if self.name == "left":
            base = half + drift
        elif self.name == "right":
            base = half - drift
        else:
            base = 6.0
            if 8.0 < (t % 30.0) < 12.0:                         # periodic obstacle
                base = 1.0
        grid = np.full((g, g), base) + np.random.normal(0, 0.02, (g, g))
        # synthetic tilt -> linear gradient across columns (exercise algo #2)
        tilt = 0.06 * math.sin(t * 0.8)
        grid += tilt * (np.arange(g) - g / 2)[None, :] * 0.1
        return np.clip(grid, 0.05, 8.0)


class LidarArray:
    def __init__(self, simulate: bool | None = None):
        self.simulate = (not HW_AVAILABLE) if simulate is None else simulate
        self.mode = DISARM
        names = list(config.LIDAR["addresses"].keys())
        if self.simulate:
            self._sensors = {n: _SimSensor(n) for n in names}
        else:                                          # pragma: no cover
            self._sensors = {}
            for n, addr in config.LIDAR["addresses"].items():
                s = vl53l7cx.VL53L7CX(bus=config.LIDAR["i2c_bus"], address=addr)
                s.set_resolution(config.LIDAR["grid"] ** 2)
                s.start_ranging()
                self._sensors[n] = s
        self._fused_lateral = 0.0                      # Weighted-LPF state

    # ------------------------------------------------------------------ raw
    def read(self, name) -> np.ndarray:
        s = self._sensors[name]
        grid = s.ranging_data() if self.simulate else \
            np.asarray(s.get_ranging_data(), float).reshape(
                config.LIDAR["grid"], -1) / 1000.0
        for r, c in config.LIDAR["deadzone_cells"]:    # rotor-arm deadzones
            grid[r, c] = np.nan                        # simple DSP rejection
        return grid

    # -------------------------------------------------- algorithm 1: peaks
    @staticmethod
    def peak_threshold(grid: np.ndarray) -> float | None:
        """Closest cell, accepted only if any 8-neighbour corroborates it
        within 30 cm -- rejects single-cell sunlight glint."""
        g = np.where(np.isnan(grid), np.inf, grid)
        idx = np.unravel_index(np.argmin(g), g.shape)
        d = g[idx]
        if not np.isfinite(d):
            return None
        r, c = idx
        nb = g[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2].flatten()
        nb = nb[np.isfinite(nb)]
        corroborated = np.sum(np.abs(nb - d) < 0.30) >= 2   # itself + 1 adjacent
        return float(d) if corroborated else None

    # ----------------------------------------------- algorithm 2: gradient
    @staticmethod
    def wall_gradient(grid: np.ndarray):
        """Least-squares plane over the mesh => tilt-compensated wall distance
        (plane offset at boresight) + gradient magnitude."""
        g = config.LIDAR["grid"]
        ys, xs = np.mgrid[0:g, 0:g]
        m = np.isfinite(grid)
        A = np.column_stack([xs[m], ys[m], np.ones(m.sum())])
        coef, *_ = np.linalg.lstsq(A, grid[m], rcond=None)
        a, b, c0 = coef
        center = a * (g - 1) / 2 + b * (g - 1) / 2 + c0
        return float(center), float(math.hypot(a, b))

    # ----------------------------------------------------------- composite
    def scan(self) -> dict:
        out = {"mode": self.mode}
        for name in self._sensors:
            grid = self.read(name)
            out[name] = {
                "grid": grid,
                "peak_m": self.peak_threshold(grid),
                "wall_m": self.wall_gradient(grid)[0],
            }
        f = out["front"]["peak_m"]
        out["forward_obstacle"] = (self.mode != DISARM and f is not None
                                   and f < config.LIDAR["obstacle_thresh_m"])
        mins = [v["peak_m"] for k, v in out.items()
                if isinstance(v, dict) and v.get("peak_m") is not None]
        out["failsafe_trip"] = (self.mode == ARM_FS and mins
                                and min(mins) < config.LIDAR["failsafe_thresh_m"])
        # lateral innovation: +ve = drifted right of corridor centreline
        out["lateral_innovation_m"] = (
            out["left"]["wall_m"] - out["right"]["wall_m"]) / 2.0
        return out

    # --------------------------------------- Weighted-LPF fusion (Innov. 1)
    def fuse_lateral(self, vision_offset_m: float, scan: dict) -> float:
        w_v = config.LIDAR["lpf_vision_weight"]
        w_l = config.LIDAR["lpf_lidar_weight"]
        target = w_v * vision_offset_m + w_l * scan["lateral_innovation_m"]
        self._fused_lateral += 0.30 * (target - self._fused_lateral)   # LPF
        return self._fused_lateral

    def set_mode(self, mode: str):
        assert mode in (DISARM, ARM, ARM_FS)
        self.mode = mode
