# 🍾 QC-Vision: Real-Time Bottle Defect Detection System

> **Capstone Project — Industrial Quality Control using AI + Edge Computing**
>
> A real-time bottle inspection system that runs on a **Raspberry Pi 4**, uses a
> **MobileNetV2** deep-learning model to classify bottles as ✅ GOOD / 🚫 BAD / ❓ OTHER,
> and streams live results to a **Django web dashboard** over the local network.

---

## 📸 System Overview

```
┌─────────────┐   MJPEG    ┌────────────────────┐   HTTP     ┌──────────────┐
│  USB Camera │ ─────────▶ │  Raspberry Pi 4    │ ─────────▶ │  Browser     │
│  (V4L2)     │            │  Django + TFLite   │            │  Dashboard   │
└─────────────┘            │  SQLite Database   │            └──────────────┘
                           └────────────────────┘
                                    │
                              Edge-to-Fog
                              Architecture
```

| Component | Technology |
|-----------|-----------|
| Edge Device | Raspberry Pi 4 Model B |
| AI Model | MobileNetV2 (Transfer Learning) |
| Inference Engine | TensorFlow Lite (`.tflite`) on Pi / Keras on Server |
| Web Backend | Django 4.2 |
| Database | SQLite (auto-resets per session) |
| Camera Driver | V4L2 + MJPEG forced encoding |
| Deployment | Docker Compose |

---

## 🗂️ Project Structure

```
bottle_project/
├── core/                         # Django project root
│   ├── core/                     # Django settings & config
│   │   ├── settings.py
│   │   ├── urls.py
│   │   ├── utils.py
│   │   ├── wsgi.py
│   │   └── asgi.py
│   │
│   ├── detector/                 # Main Django app
│   │   ├── migrations/           # Database migrations
│   │   ├── templates/
│   │   │   ├── detector/
│   │   │   │   ├── dashboard.html   # Real-time monitoring UI
│   │   │   │   └── register.html
│   │   │   └── registration/
│   │   │       └── login.html
│   │   ├── admin.py              # Django admin configuration
│   │   ├── apps.py               # App startup + camera init
│   │   ├── camera.py             # 🔑 Core: CV + AI inference engine
│   │   ├── models.py             # DefectLog & SystemConfig models
│   │   ├── urls.py               # App URL routing
│   │   └── views.py              # Web views + video stream
│   │
│   ├── ml_model/                 # (gitignored) ML training + model files
│   │   ├── train.py              # MobileNetV2 training script
│   │   ├── test_camera.py        # Camera hardware test
│   │   ├── test_industrial_stable.py
│   │   ├── test_smart_focus.py
│   │   └── dataset/              # Training images (BAD / GOOD / OTHER)
│   │
│   ├── manage.py                 # Django management CLI
│   ├── requirements.txt          # Server/dev dependencies
│   ├── requirements-pi.txt       # Raspberry Pi dependencies (TFLite)
│   └── OPERATIONS.md             # Full SOP for daily operation
│
└── README.md                     # ← You are here
```

---

## 🧠 How It Works

### 1. Object Detection (Smart Focus)
The camera feed is processed frame-by-frame using OpenCV:
- **Canny edge detection** isolates the largest object in frame
- A bounding box is drawn, and the ROI (Region of Interest) is cropped
- If the object is too small → prompts operator to **"MOVE CLOSER"**

### 2. AI Classification
The cropped ROI is resized to `224×224` and passed through the model:

| Prediction | Meaning |
|------------|---------|
| Class 0 — BAD | Defective bottle (broken cap, deformed body, etc.) |
| Class 1 — GOOD | Passes quality check |
| Class 2 — OTHER | Background / non-bottle object |

A **sliding buffer of 15 frames** is used to stabilise predictions — a defect only triggers an alert after **4+ consecutive BAD frames**.

### 3. Defect Logging
When a defect is confirmed:
- A `DefectLog` record is written to SQLite (with timestamp + confidence score)
- A 5-second cooldown prevents duplicate logs for the same bottle
- Logs are available for **CSV export** from the dashboard

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- A USB webcam (or Raspberry Pi Camera)
- *(For Pi deployment)* Raspberry Pi 4 with Docker installed

### 1. Clone the Repository

```bash
git clone https://github.com/Awesome-sanyam/my-bottle_detection-project.git
cd my-bottle_detection-project/core
```

### 2. Set Up Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
# On a development machine / server:
pip install -r requirements.txt

# On Raspberry Pi:
pip install -r requirements-pi.txt
```

### 4. Add Your ML Model

The model files are not tracked in Git (they are large binary files).
Place your trained model at:

```
core/ml_model/bottle_model_multiclass.keras   # for Keras / server
core/ml_model/bottle_model_quantized.tflite   # for TFLite / Raspberry Pi
```

> See [`ml_model/train.py`](core/ml_model/train.py) to train your own model.

### 5. Apply Migrations & Create Admin User

```bash
python manage.py migrate
python manage.py createsuperuser
```

### 6. Run the Development Server

```bash
python manage.py runserver 0.0.0.0:8000
```

Open `http://localhost:8000` in your browser.

---

## 🍓 Raspberry Pi Deployment

See [OPERATIONS.md](core/OPERATIONS.md) for the full SOP, including:
- Docker Compose startup / shutdown
- Code sync from Mac → Pi (`rsync`)
- Dashboard access over LAN
- Admin panel configuration
- Troubleshooting and log viewing

---

## 🤖 Training Your Own Model

```bash
cd core/ml_model

# Organize dataset:
# dataset/
#   ├── BAD/     ← images of defective bottles
#   ├── GOOD/    ← images of good bottles
#   └── OTHER/   ← background / non-bottle images

python train.py
```

Training uses **MobileNetV2** (ImageNet pre-trained) with two-phase transfer learning:
1. **Phase 1** — Train the classification head only (10 epochs)
2. **Phase 2** — Fine-tune the last 50 layers (25 epochs, early stopping)

---

## 📊 Key Features

- ✅ **Real-time MJPEG video stream** with overlay annotations
- ✅ **3-class classification**: BAD / GOOD / OTHER
- ✅ **Sliding window vote buffer** — eliminates single-frame false positives
- ✅ **Auto-logging** to SQLite with cooldown de-duplication
- ✅ **CSV export** of defect logs from the dashboard
- ✅ **Admin panel** to tune confidence threshold without code changes
- ✅ **Edge-compatible** — runs on Pi with TFLite, or on server with Keras
- ✅ **User authentication** — login-gated dashboard

---

## 🔒 Security Notes

> [!WARNING]
> `settings.py` currently contains a hardcoded `SECRET_KEY` and `DEBUG = True`.
> **Before production deployment:**
> - Move `SECRET_KEY` to an environment variable
> - Set `DEBUG = False`
> - Set `ALLOWED_HOSTS` to your specific domain/IP

---

## 🧪 Running Tests

```bash
# Django unit tests
python manage.py test

# Camera hardware test (requires webcam)
python ml_model/test_camera.py

# Industrial stable detection test
python ml_model/test_industrial_stable.py
```

---

## 📁 Files NOT Tracked in Git

The following are intentionally excluded (see `.gitignore`):

| Path | Reason |
|------|--------|
| `ml_model/*.keras`, `*.tflite`, `*.h5` | Large binary files (use Git LFS or external storage) |
| `ml_model/dataset/` | Training images (too large) |
| `.venv/` | Virtual environment |
| `db.sqlite3` | Runtime database |
| `.DS_Store` | macOS system file |

---

## 👨‍💻 Author

**Sanyam Gehlot**
Capstone Project — 2026
GitHub: [@Awesome-sanyam](https://github.com/Awesome-sanyam)
