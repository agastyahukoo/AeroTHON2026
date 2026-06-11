"""
winch.py -- Single-servo winch + gravity-release hook per Design Report 3.3.

Mechanism: TowerPro MG90S continuous-rotation servo, 30 mm spool, 0.8 mm nylon
monofilament (5.5 m loaded), nitrile pretensioner cord, PETG gravity hook.

  Lowering rate : 4.7 cm/s @ 30 rpm  ->  106 s for the full 5 m descent
  Release       : payload ground contact -> line slack -> hook self-releases
                  under its own weight (no second servo)
  Stow          : retract + pretension overwind so the elastic cord clamps the
                  payload into the bay (sway / unloading mitigation)

Sequence wired into FSM S5 PAYLOAD DELIVERY:
  spool-out (106 s) -> settle 15 s (slack confirms gravity release)
  -> retract -> settle 15 s -> pretension overwind.

On the Pi 5 the servo runs on hardware PWM; elsewhere a timed simulator with
the identical state/progress interface drives the GCS payload-lowering demo.
"""

from __future__ import annotations

import threading
import time

try:
    from . import config
except ImportError:
    import config

try:                                                   # pragma: no cover
    from rpi_hardware_pwm import HardwarePWM
    HW_AVAILABLE = True
except Exception:
    HardwarePWM = None
    HW_AVAILABLE = False

IDLE, LOWERING, SETTLE_OUT, RETRACTING, SETTLE_IN, PRETENSION, STOWED, RELEASED \
    = ("IDLE", "LOWERING", "SETTLE-OUT", "RETRACTING", "SETTLE-IN",
       "PRETENSION", "STOWED", "RELEASED")


class Winch:
    def __init__(self, simulate: bool | None = None, time_scale: float = 1.0):
        """time_scale > 1 accelerates the timeline for bench/GCS demos."""
        self.simulate = (not HW_AVAILABLE) if simulate is None else simulate
        self.time_scale = time_scale
        self.state = IDLE
        self.deployed_m = 0.0
        self.hook_released = False
        self.current_ma = 0
        self._stop = threading.Event()
        self._thread = None
        if not self.simulate:                          # pragma: no cover
            self._pwm = HardwarePWM(pwm_channel=0, hz=50)
            self._pwm.start(self._duty(config.WINCH["pwm_neutral_us"]))

    # ------------------------------------------------------------- pwm io
    @staticmethod
    def _duty(us):
        return us / 20000.0 * 100.0

    def _drive(self, us):
        if not self.simulate:                          # pragma: no cover
            self._pwm.change_duty_cycle(self._duty(us))
        self.current_ma = 200 if us != config.WINCH["pwm_neutral_us"] else 20

    # ----------------------------------------------------------- sequence
    def deliver(self, blocking=False):
        """Full FSM-S5 delivery cycle in a worker thread."""
        if self._thread and self._thread.is_alive():
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._cycle, daemon=True)
        self._thread.start()
        if blocking:
            self._thread.join()
        return True

    def _cycle(self):
        w = config.WINCH
        rate = w["lower_rate_cm_s"] / 100.0 * self.time_scale     # m/s
        # 1. spool out -------------------------------------------------
        self.state, self.hook_released = LOWERING, False
        self._drive(w["pwm_lower_us"])
        while self.deployed_m < w["descent_len_m"] and not self._stop.is_set():
            time.sleep(0.05)
            self.deployed_m = min(self.deployed_m + rate * 0.05,
                                  w["descent_len_m"])
        self._drive(w["pwm_neutral_us"])
        # 2. settle: ground contact -> slack -> gravity release ---------
        self.state = SETTLE_OUT
        self._sleep(w["settle_delay_s"])
        self.hook_released = True
        self.state = RELEASED
        self._sleep(1.0)
        # 3. retract -----------------------------------------------------
        self.state = RETRACTING
        self._drive(w["pwm_retract_us"])
        while self.deployed_m > 0.0 and not self._stop.is_set():
            time.sleep(0.05)
            self.deployed_m = max(self.deployed_m - rate * 0.05, 0.0)
        self._drive(w["pwm_neutral_us"])
        # 4. settle + pretension overwind --------------------------------
        self.state = SETTLE_IN
        self._sleep(w["settle_delay_s"])
        self.state = PRETENSION
        self._drive(w["pwm_retract_us"])
        self.current_ma = w["stall_current_ma"]        # elastic cord loads servo
        self._sleep(w["pretension_overwind_s"])
        self._drive(w["pwm_neutral_us"])
        self.state = STOWED

    def _sleep(self, s):
        t_end = time.monotonic() + s / self.time_scale
        while time.monotonic() < t_end and not self._stop.is_set():
            time.sleep(0.05)

    def abort(self):
        self._stop.set()
        self._drive(config.WINCH["pwm_neutral_us"])
        self.state = IDLE

    def reset(self):
        self.abort()
        self.deployed_m = 0.0
        self.hook_released = False
        self.state = IDLE

    def status(self) -> dict:
        w = config.WINCH
        return {"state": self.state,
                "deployed_m": round(self.deployed_m, 3),
                "deployed_pct": round(100 * self.deployed_m /
                                      w["descent_len_m"], 1),
                "hook_released": self.hook_released,
                "current_ma": self.current_ma,
                "rate_cm_s": w["lower_rate_cm_s"],
                "full_descent_s": w["full_descent_s"]}
