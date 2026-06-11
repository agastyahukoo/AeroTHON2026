"""
AT2026RC67 / MPSTME AeroXperts -- SAE AeroTHON 2026
====================================================
config.py -- Single source of truth for every mission parameter.

All values trace directly to the AT2026 Design Report:
  - Winch: TowerPro MG90S 360deg, 30 mm spool, 30 rpm  -> 4.7 cm/s, 106 s / 5 m
  - Cameras: 2x Sony IMX708 (forward 120deg FOV, downward 76deg FOV)
  - LiDAR: 3x ST VL53L7CX (8x8 ToF), ~270deg frontal coverage
  - FC: Radiolink Pixhawk 2.4.8 over MAVLink/UART (TELEM2 <-> Pi GPIO14/15)
  - Mission: FM2 per rulebook (QR @5m, corridor @3m, ID @10m, deliver @5m)
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# MAVLink / Flight controller (Section 1.9, 3.1)
# ----------------------------------------------------------------------------
MAVLINK = {
    # On the Pi 5: Pixhawk TELEM2 -> GPIO14/15 (/dev/serial0).
    "connection": os.environ.get("AX_MAVLINK", "/dev/serial0"),
    "baud": 57600,
    "system_id": 255,
    "heartbeat_timeout_s": 3.0,      # OBC watchdog -> LAND failsafe (report 1.1)
    # MAV_CMD_DO_GUIDED_LIMITS (Safety section A): anti-flyaway hard limits
    "guided_limits": {
        "timeout_s": 900,            # 15 min mission window
        "min_alt_m": 0.5,
        "max_alt_m": 18.0,           # hard ceiling, below event ceiling
        "max_horiz_m": 250.0,        # circular fence in addition to official fence
    },
}

# ----------------------------------------------------------------------------
# Mission altitudes & motion (FM2 Operation, rulebook + FSM Table 19)
# ----------------------------------------------------------------------------
MISSION = {
    "takeoff_alt_m": 5.0,            # S1: scan start QR from 5 m
    "scan_forward_m": 1.0,           # S1: move forward ~1 m
    "corridor_alt_m": 3.0,           # S2/S3: descend to 10 ft for corridor
    "corridor_width_m": 3.5,
    "corridor_min_len_m": 12.0,      # S3 exit: dist from CORRIDOR_START_WPT > 12 m
    "delivery_search_alt_m": 10.0,   # S4: ascend to 10 m to identify QR
    "delivery_alt_m": 5.0,           # S5: descend to 5 m, lower payload
    "sideslip_step_m": 0.25,         # B: forward offset per attempt
    "sideslip_avoid_step_m": 0.10,   # B: lateral step per attempt
    "nudge_step_m": 0.25,            # S5 positioning increments
    "fine_nudge_m": 0.10,            # vision-augmented sub-1m corrections
    "heading_trim_deg": 15.0,        # B/E: corridor wall alignment limit
    "rtl_disarm_delay_s": 30.0,      # S8
    "time_limit_s": 900.0,
}

# ----------------------------------------------------------------------------
# Winch / gravity hook (Section 3.3)
# ----------------------------------------------------------------------------
WINCH = {
    "gpio_pin": 12,                  # hardware PWM channel on Pi 5
    "spool_diameter_mm": 30.0,
    "rpm": 30.0,
    "lower_rate_cm_s": 4.7,          # = pi*30mm*30rpm/60
    "descent_len_m": 5.0,            # full spool-out
    "full_descent_s": 106.0,         # 5 m / 4.7 cm/s
    "settle_delay_s": 15.0,          # FSM: delay 15 s, retract, delay 15 s
    "pretension_overwind_s": 1.5,    # nitrile-cord pretensioner overwind
    "pwm_neutral_us": 1500,
    "pwm_lower_us": 1700,            # continuous-rotation MG90S, ~30 rpm
    "pwm_retract_us": 1300,
    "stall_current_ma": 400,         # validated by PWM sweep (Fig 19)
}

# ----------------------------------------------------------------------------
# Vision (Sections 1.9, 3.2)
# ----------------------------------------------------------------------------
VISION = {
    "forward_cam_index": 0,
    "down_cam_index": 1,
    "frame_w": 1280,
    "frame_h": 720,
    "target_fps": 15,                # reliable 10-15 fps minimum (report 3.2)
    # --- Raspberry Pi 5 performance budget -------------------------------
    # OPTIONAL processing downscale. 1.0 = full resolution (DEFAULT and
    # recommended: targets are on the ground ~10 m below the down camera,
    # so pixels-on-target are precious). 0.75 / 0.5 trade detection range
    # for CPU headroom; results are always rescaled to full-frame coords.
    "proc_scale": 1.0,
    "stream_width": 960,             # MJPEG downlink size / JPEG quality
    "stream_quality": 70,
    "cv_threads": 3,                 # leave one A76 core for FSM/MAVLink/server
    # QR pipeline: OpenCV primary, pyzbar fallback, time-based toggling
    "qr_toggle_period_s": 1.0,
    "qr_persistence_frames": 3,      # same string across N frames -> accepted
    # Banner (green AeroTHON logo) HSV candidate gate -- tuned over grass
    "banner_hsv_lo": (40, 60, 50),
    "banner_hsv_hi": (85, 255, 255),
    "banner_min_area_px": 1500,
    "banner_aspect_range": (1.2, 3.5),   # banner is wide rectangle
    "banner_rectangularity_min": 0.70,   # contourArea / boundingRect area
    "banner_persistence_frames": 4,
    # Red restricted-zone segmentation (two hue lobes)
    "red_hsv_lo1": (0, 110, 70),  "red_hsv_hi1": (10, 255, 255),
    "red_hsv_lo2": (170, 110, 70), "red_hsv_hi2": (180, 255, 255),
    "red_min_area_px": 2500,
    "red_roi_frac": 0.45,            # traversal-zone ROI (centre band) fraction
    # Target centering (S5): QR centroid within tolerance for N frames
    "center_tol_frac": 0.06,         # of frame width
    "center_persistence_frames": 5,
    # Digital image stabilisation (Innovation 3): bounded affine correction
    "dis_max_shift_px": 80,
}

# ----------------------------------------------------------------------------
# LiDAR -- 3x VL53L7CX (Sections 1.9, 3.2 Corridor Vision-ToF Fusion)
# ----------------------------------------------------------------------------
LIDAR = {
    "i2c_bus": 1,
    "addresses": {"left": 0x29, "front": 0x2A, "right": 0x2B},
    "grid": 8,                                   # 8x8 multizone mesh
    "fov_deg": 45.0,
    "obstacle_thresh_m": 1.2,        # forward trigger -> sideslip avoidance
    "failsafe_thresh_m": 0.6,        # ARM-FS proximity -> RTL/LAND
    "wall_buffer_m": 0.75,           # side wall safe buffer in 3.5 m corridor
    "deadzone_cells": [(0, 0), (0, 7)],          # rotor-arm convergence cells
    "lpf_vision_weight": 0.35,       # Weighted-LPF fusion (Innovation 1)
    "lpf_lidar_weight": 0.65,
}

# ----------------------------------------------------------------------------
# Detection logging -- every confirmed detection is written to a timestamped
# JSONL file (QR entries carry decoded text + pixel location + GPS fix)
# ----------------------------------------------------------------------------
LOGGING = {
    "dir": os.path.join(BASE_DIR, "logs"),
    "detections_file": "detections_%Y%m%d.jsonl",    # strftime pattern
    "mission_log": os.path.join(BASE_DIR, "logs", "mission_log.json"),
}

# ----------------------------------------------------------------------------
# Geofencing (Safety section B: dual safety fences + mission fence)
# ----------------------------------------------------------------------------
GEOFENCE = {
    "file": os.path.join(BASE_DIR, "mission_geofence.json"),
    "obc_margin_m": 4.0,             # OBC fence is TIGHTER than official by this
    "proximity_warn_m": 3.0,         # reverse to previous WPT inside this margin
}

# ----------------------------------------------------------------------------
# Airframe specs (Appendix A1) -- consumed by GCS / diagnostics UI
# ----------------------------------------------------------------------------
SPECS = {
    "Team":              "MPSTME AeroXperts (AT2026RC67)",
    "Configuration":     "H-frame quadrotor",
    "MTOW":              "1.70 kg target / <2.00 kg limit",
    "Payload":           "100 g, controlled lowering (gravity hook)",
    "Motors":            "4x T-Motor AT2312 1150 KV",
    "Propellers":        "GEMFAN APC 9x4.5MR-B4",
    "ESC":               "4x FlamebackTech 45 A (BLHeli_S)",
    "Battery":           "4S2P Li-Ion 14.8 V 8.4 Ah 11C (Molicel 21700)",
    "Flight controller": "Radiolink Pixhawk 2.4.8 (ArduCopter 4.6)",
    "OBC":               "Raspberry Pi 5 + Hailo AI HAT+ 26 TOPS",
    "Cameras":           "2x Sony IMX708 (forward 120deg / down 76deg)",
    "ToF sensors":       "3x ST VL53L7CX (270deg frontal)",
    "Winch":             "TowerPro MG90S 360deg, 0.8 mm nylon monofilament",
    "Wheelbase":         "488 mm diagonal",
    "Prop clearance":    "34 mm minimum / 45 mm tip-to-tip",
    "Landing gear":      "250 mm lateral x 280 mm longitudinal",
    "RC link":           "Radiolink AT9S Pro + R12DS (SBUS), >1.5 km tested",
    "Datalink":          "4G/LTE (7semi EC200U) 720p30 <0.5 s, analog FPV fallback",
    "Endurance":         "13 min hover w/ payload, 17 min w/o (80% DoD)",
}

# ----------------------------------------------------------------------------
# Tuning overrides -- written by the GCS diagnostics app ("Save tuning to
# flight config"). Loaded last so field-tuned HSV ranges / areas / tolerances
# override the defaults above and travel with the exported onboard package.
# ----------------------------------------------------------------------------
TUNABLE_KEYS = (
    "banner_hsv_lo", "banner_hsv_hi", "banner_min_area_px",
    "banner_aspect_range", "banner_rectangularity_min",
    "red_hsv_lo1", "red_hsv_hi1", "red_hsv_lo2", "red_hsv_hi2",
    "red_min_area_px", "red_roi_frac", "center_tol_frac", "proc_scale",
)
TUNING_FILE = os.path.join(BASE_DIR, "tuning_overrides.json")


def load_tuning_overrides():
    if not os.path.exists(TUNING_FILE):
        return {}
    import json
    try:
        with open(TUNING_FILE) as f:
            data = json.load(f)
    except Exception:
        return {}
    applied = {}
    for k, val in data.items():
        if k in TUNABLE_KEYS:
            VISION[k] = tuple(val) if isinstance(val, list) else val
            applied[k] = VISION[k]
    return applied


def save_tuning_overrides():
    import json
    with open(TUNING_FILE, "w") as f:
        json.dump({k: VISION[k] for k in TUNABLE_KEYS}, f, indent=2)
    return TUNING_FILE


load_tuning_overrides()
