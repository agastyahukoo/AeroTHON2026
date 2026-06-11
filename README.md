<div align="center">

<img src="https://img.shields.io/badge/Raspberry%20Pi%205-8GB-A22846?style=for-the-badge&logo=raspberrypi&logoColor=white" alt="Raspberry Pi 5">
<img src="https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
<img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/MAVLink-ArduCopter%204.6-2C2C2C?style=for-the-badge" alt="MAVLink">

# FM2 Autonomy Stack + GCS Diagnostics

**SAE AeroTHON 2026 · Rotorcraft Systems Challenge (UAS)**

Mission code that flies on a Raspberry Pi 5, and a black-glass GCS app to
test, refine and fine-tune every pipeline before it does.

</div>

---

## Layout

```
aeroxperts/
├── onboard/                 🍓  LIVE — deploy this folder to the Pi 5
│   ├── mission.py              flight entry point (+ component demos)
│   ├── fsm.py                  mission state machine S0–S8 + SX (Table 19)
│   ├── vision.py               QR · banner · red-zone · centering pipelines
│   ├── lidar.py                3× VL53L7CX — peak-threshold + wall-gradient
│   ├── mavlink_io.py           Pixhawk 2.4.8 link (GUIDED, OFFSET_NED, fences)
│   ├── geofence.py             dual safety fence + S4 search grid
│   ├── winch.py                MG90S winch · gravity hook · 4.7 cm/s
│   ├── detlog.py               timestamped detection log (JSONL + GPS)
│   ├── config.py               every parameter, one file
│   ├── mission_geofence.json   written by the GCS planning page
│   └── logs/                   detections_YYYYMMDD.jsonl · mission_log.json
├── demo_server.py           🖥  GCS bridge — imports onboard/, adds zero logic
├── webapp.html                  GCS diagnostics UI (single file, Electron-ready)
└── README.md
```

The split is strict: **`onboard/` is the mission**, `demo_server.py` +
`webapp.html` are the test bench. The GCS never re-implements anything — it
calls the exact modules that fly.

---

## 🍓 Raspberry Pi 5 — setup in four commands

```bash
sudo apt install -y python3-opencv python3-picamera2 python3-numpy libzbar0
pip install -r onboard/requirements.txt --break-system-packages
sudo raspi-config nonint do_serial_hw 0          # enable /dev/serial0 (Pixhawk TELEM2)
echo "dtoverlay=pwm,pin=12,func=4" | sudo tee -a /boot/firmware/config.txt   # winch PWM
```

Run it:

```bash
cd onboard
python3 mission.py                    # ▶ live FM2 (cameras, LiDAR, Pixhawk)
python3 mission.py --dry-run          # FSM on SIM MAVLink/LiDAR — real vision
python3 mission.py --demo qr          # also: banner · redzone · lidar · winch · geofence
```

Test bench (same Pi, or any laptop):

```bash
python3 demo_server.py                # → http://<pi>:8742
```

---

## Built for the Pi 5

| Optimisation | What it does |
|---|---|
| **Picamera2 / libcamera capture** | CSI IMX708s go through the hardware ISP — no FFmpeg copy chain, minimal CPU. V4L2 fallback for bench USB cams. |
| **Latest-frame grabber threads** | One thread per camera holds only the newest frame: pipelines never block on capture and never process a stale buffer. |
| **`cv_threads = 3`** | OpenCV is capped to 3 of the 4 A76 cores; the FSM, MAVLink RX and server keep a core to themselves. |
| **Lazy preprocessing** | The CLAHE + adaptive-threshold copy is built **only when the raw frame fails to decode**; CLAHE objects and morphology kernels are cached, never re-created per frame. |
| **Optional downscale (`PROC SCALE`)** | Default **1.00 — full resolution** (ground targets seen from 10 m need every pixel). 0.75 / 0.5 are selectable in the GCS when CPU headroom matters; geometry is always rescaled to full-frame coordinates. No frame skipping — detection runs **every frame**. |
| **Idle-aware streaming** | With no GCS viewer connected the loop drops to 2 Hz and skips JPEG encode entirely; the MJPEG downlink is resized to 960 px / Q70 with a live **FPS + scale overlay** on every stream. |
| **Real input only** | No synthetic scenes, no simulated detections — anywhere. With no source attached the stream shows a NO INPUT card that cannot produce a detection; in the FSM, camera loss is a failsafe, a missed target is a timeout → SX, exactly as in flight. |

---

## Detection logging

Every confirmed detection is appended to
`onboard/logs/detections_YYYYMMDD.jsonl` — one timestamped JSON object per
line, de-duplicated over a 3 s window. **QR entries always carry the decoded
text, the pixel location, and the GPS fix** taken from MAVLink at the moment
of detection:

```json
{"ts": "2026-06-12T10:41:03.214+05:30", "type": "qr",
 "data": "DELIVERY:ZONE-B", "decoder": "opencv",
 "px": [641.2, 388.7],
 "gps": {"lat": 19.1076120, "lon": 72.8370410, "alt_m": 9.98},
 "state": "S4", "role": "delivery_match"}
```

Banner and red-zone events log bbox + centroid + GPS the same way. The file
is viewable and downloadable from the GCS **QR DECODE** tab, and the FSM
writes a full `mission_log.json` next to it after every flight.

---

## GCS diagnostics app

Keyboard: <kbd>1</kbd>–<kbd>5</kbd> switch pages, <kbd>S</kbd> saves a full-size
snapshot of the active pipeline output (also available as a button on every
stream — works even when the stream is idle). All saves and input changes
confirm with toast notifications; the nav rail shows a live LINK indicator.


| Page | Purpose |
|---|---|
| **Main menu** | Typed *Welcome, \<operator\>* greeting, live system-status card (source / FPS / scale / decoders / fence / mission code), full airframe spec sheet + architecture summary. |
| **FM1 · FPV** | Raw camera stream, no CV attached. |
| **FM2 · Diagnostics** | Per-feature tests on the flight code, all driven by a **real test input — camera, image or video** (drag-and-drop, pausable video, `PROC SCALE` selector): |
| | · **QR DECODE** — dual-decoder pipeline + detection-log viewer/download |
| | · **TARGET ID** — first QR sets the mission code; matching boards box green, others amber |
| | · **COLOUR ISOLATION** — colour picker + HSV/area/rectangularity tuning, with **Apply to flight config**: writes straight into `config.VISION`, takes effect live in the flight pipelines, persists to `tuning_overrides.json` and ships in the export |
| | · **BANNER DETECT** — S2 hybrid pipeline with persistence gate |
| | · **PAYLOAD DELIVERY** — centroid centering, 10 m → 5 m descent, real winch cycle |
| | · **FSM DRY RUN** — `fsm.py` end-to-end on SIM MAVLink/LiDAR, **real vision** |
| **Settings** | Operator name (typed welcome on the main menu), default camera, stream quality and default processing scale — stored server-side in `gcs_settings.json`, applied live and across restarts. |
| **Mission plan · Export** | Dark CARTO/OpenStreetMap map. Draw by clicking **or type GPS coordinates** (single point or paste the coordinate list handed out on site) into the selected layer — official fence, delivery zone, corridor, **starting point**. Saves to `mission_geofence.json`; exports the ready-to-deploy `onboard/` ZIP. |

---

## Safety chain (Design Report §4)

Pixhawk geofence → OBC point-in-polygon fence (−4 m inset) → mission fence
bounding the S4 grid · `MAV_CMD_DO_GUIDED_LIMITS` (18 m ceiling / 250 m
radius / 15 min) · LiDAR **ARM-FS** proximity → LAND · FC-heartbeat watchdog ·
**camera-loss failsafe** · battery / EKF / FENCE_STATUS failsafes → RTL or LAND.
