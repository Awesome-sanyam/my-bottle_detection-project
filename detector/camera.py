import cv2
import os
import sys
import platform
import numpy as np
import threading
import time
from collections import deque
from django.conf import settings
from .models import DefectLog

# ── OS Detection ─────────────────────────────────────────────────────────────
IS_LINUX   = platform.system() == 'Linux'    # Raspberry Pi / production
IS_MAC     = platform.system() == 'Darwin'   # Development machine
IS_WINDOWS = platform.system() == 'Windows'

# ── Model Path Resolution ────────────────────────────────────────────────────
# Priority 1: quantized TFLite (Pi production)
# Priority 2: full .tflite (Pi fallback)
# Priority 3: multiclass Keras (server/Mac)
_ML_DIR = os.path.join(settings.BASE_DIR, 'ml_model')

TFLITE_QUANTIZED = os.path.join(_ML_DIR, 'bottle_model_quantized.tflite')
TFLITE_STANDARD  = os.path.join(_ML_DIR, 'bottle_model.tflite')
KERAS_MODEL      = os.path.join(_ML_DIR, 'bottle_model_multiclass.keras')
KERAS_V2_MODEL   = os.path.join(_ML_DIR, 'bottle_model_v2.keras')

# Pick the best available model
if os.path.exists(TFLITE_QUANTIZED):
    TFLITE_MODEL_PATH = TFLITE_QUANTIZED
    USE_TFLITE_MODE   = True
elif os.path.exists(TFLITE_STANDARD):
    TFLITE_MODEL_PATH = TFLITE_STANDARD
    USE_TFLITE_MODE   = True
else:
    TFLITE_MODEL_PATH = None
    USE_TFLITE_MODE   = False

# Pick best Keras model
if os.path.exists(KERAS_MODEL):
    KERAS_MODEL_PATH = KERAS_MODEL
elif os.path.exists(KERAS_V2_MODEL):
    KERAS_MODEL_PATH = KERAS_V2_MODEL
else:
    KERAS_MODEL_PATH = None

# ── TensorFlow / TFLite Import ───────────────────────────────────────────────
Interpreter = None
tf = None

try:
    import tensorflow as _tf
    tf = _tf
    if USE_TFLITE_MODE:
        Interpreter = tf.lite.Interpreter
        print("⚡ PLATFORM: EDGE DEVICE (TFLite via Full TF)")
    else:
        print("🖥️ PLATFORM: HIGH-POWER SERVER (Keras)")
except ImportError:
    try:
        import tflite_runtime.interpreter as _tflite
        Interpreter = _tflite.Interpreter
        print("⚡ PLATFORM: EDGE DEVICE (TFLite Runtime)")
    except ImportError:
        print("⚠️ WARNING: No TensorFlow/TFLite found — running in NO-MODEL mode")


# ── Placeholder Frame Generator ──────────────────────────────────────────────
def _make_offline_frame(msg="📷 CAMERA OFFLINE", sub="Connect a USB camera & restart"):
    """Generates a dark info frame returned when no camera is available."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (18, 18, 24)                     # dark background
    # draw a border
    cv2.rectangle(frame, (10, 10), (630, 470), (40, 40, 55), 2)
    # camera-off icon (simple X cross)
    cx, cy = 320, 180
    cv2.circle(frame, (cx, cy), 60, (60, 60, 80), 2)
    cv2.line(frame, (cx-42, cy-42), (cx+42, cy+42), (80, 80, 100), 3)
    cv2.line(frame, (cx+42, cy-42), (cx-42, cy+42), (80, 80, 100), 3)
    # main message
    cv2.putText(frame, msg, (50, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 220), 2, cv2.LINE_AA)
    # sub message
    cv2.putText(frame, sub, (50, 340),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 140), 1, cv2.LINE_AA)
    # status bar
    cv2.rectangle(frame, (0, 0), (640, 50), (30, 30, 40), -1)
    cv2.putText(frame, "NO SIGNAL", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (200, 200, 210), 2, cv2.LINE_AA)
    ret, jpeg = cv2.imencode('.jpg', frame)
    return jpeg.tobytes() if ret else None


class VideoCamera(object):
    """Thread-safe singleton camera + AI inference engine."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                inst = super(VideoCamera, cls).__new__(cls)
                inst.initialized   = False
                inst.video         = None
                inst.model         = None
                inst.interpreter   = None
                inst.buffer        = deque(maxlen=15)
                inst.camera_ok     = False   # True only when real camera opens
                inst.model_ok      = False   # True only when model loads
                cls._instance = inst
            return cls._instance

    # ── Initialisation ──────────────────────────────────────────────────────
    def initialize(self):
        if self.initialized:
            return

        self._open_camera()
        self._load_model()

        self.last_log_time = 0
        self.log_cooldown  = 5.0
        self.min_scan_area = 45000
        self.defect_trigger = 4
        self.initialized   = True

    def _open_camera(self):
        """Open camera with the correct backend for each OS."""
        print(f"🎥 CAMERA: Detected OS = {platform.system()}")

        if IS_LINUX:
            # Raspberry Pi / Linux — force V4L2 + MJPEG to prevent raw data crash
            print("🎥 CAMERA: Using V4L2 backend (Linux/Pi mode)")
            self.video = cv2.VideoCapture(0, cv2.CAP_V4L2)
            if self.video.isOpened():
                self.video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        else:
            # macOS / Windows — use the default backend (AVFoundation / DirectShow)
            print("🎥 CAMERA: Using default backend (Mac/Windows mode)")
            self.video = cv2.VideoCapture(0)

        if self.video and self.video.isOpened():
            self.video.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.video.set(cv2.CAP_PROP_FPS, 30)
            self.camera_ok = True
            print("✅ CAMERA: Opened successfully")
        else:
            self.camera_ok = False
            print("⚠️ CAMERA: No camera found — will stream offline frame")

    def _load_model(self):
        """Load the best available ML model."""
        if USE_TFLITE_MODE and Interpreter and TFLITE_MODEL_PATH:
            print(f"⚡ LOADING TFLITE: {TFLITE_MODEL_PATH}")
            try:
                self.interpreter = Interpreter(model_path=TFLITE_MODEL_PATH)
                self.interpreter.allocate_tensors()
                self.input_details  = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                self.model_ok = True
                print("✅ TFLITE ENGINE READY")
            except Exception as e:
                print(f"❌ TFLITE LOAD ERROR: {e}")

        elif tf and KERAS_MODEL_PATH:
            print(f"🏭 LOADING KERAS: {KERAS_MODEL_PATH}")
            try:
                self.model    = tf.keras.models.load_model(KERAS_MODEL_PATH)
                self.model_ok = True
                print("✅ KERAS ENGINE READY")
            except Exception as e:
                print(f"❌ KERAS LOAD ERROR: {e}")
        else:
            print("⚠️ MODEL: No model file found — detection disabled")

    # ── Inference ────────────────────────────────────────────────────────────
    def predict_image(self, img_array):
        """Run inference. Returns [bad_prob, good_prob, other_prob]."""
        if USE_TFLITE_MODE and self.interpreter:
            input_data = np.array(img_array, dtype=np.float32)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            return self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        elif not USE_TFLITE_MODE and self.model:
            return self.model.predict(img_array, verbose=0)[0]
        # No model: default to OTHER (safe pass-through)
        return np.array([0.0, 0.0, 1.0])

    # ── Signal Handlers ──────────────────────────────────────────────────────
    def hard_exit_handler(self, signum, frame):
        print("\n☢️ SHUTDOWN SIGNAL RECEIVED...")
        if self.video and self.video.isOpened():
            self.video.release()
        os._exit(0)

    # ── DB Logging ───────────────────────────────────────────────────────────
    def save_log_to_db(self, confidence):
        try:
            DefectLog.objects.create(confidence=float(confidence))
            print(f"📝 LOG SAVED: {confidence:.2f}")
        except Exception as e:
            print(f"❌ LOG ERROR: {e}")

    # ── Computer Vision ──────────────────────────────────────────────────────
    def get_focus_object(self, frame):
        """Return bounding rect & area of the largest object in frame."""
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur    = cv2.GaussianBlur(gray, (5, 5), 0)
        edged   = cv2.Canny(blur, 50, 150)
        kernel  = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edged, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0
        valid = [c for c in contours if cv2.contourArea(c) > 5000]
        if not valid:
            return None, 0
        best = max(valid, key=cv2.contourArea)
        return cv2.boundingRect(best), cv2.contourArea(best)

    # ── Main Frame Loop ──────────────────────────────────────────────────────
    def get_frame(self):
        """Return a JPEG-encoded frame (bytes). Never returns None — falls back to offline frame."""
        if not self.initialized:
            self.initialize()

        # ── No camera: stream the offline placeholder ──
        if not self.camera_ok or not self.video or not self.video.isOpened():
            return _make_offline_frame()

        success, frame = self.video.read()
        if not success or frame is None:
            # Try to recover
            self.video.release()
            self._open_camera()
            return _make_offline_frame("📷 CAMERA RECONNECTING...", "Signal lost — retrying")

        # ── Basic frame sanity (prevents C++ crash on bad buffers) ──
        h, w = frame.shape[:2]
        if w == 0 or h == 0 or w > 4000 or h > 4000:
            return _make_offline_frame()

        # ── Ensure 3-channel BGR ──
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # ── Object detection + classification ──
        status_msg = "WAITING..."
        box_color  = (100, 100, 100)

        if not self.model_ok:
            # No model available — just stream raw feed with a banner
            cv2.rectangle(frame, (0, 0), (w, 50), (40, 40, 60), -1)
            cv2.putText(frame, "NO MODEL — RAW FEED", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (200, 150, 0), 2, cv2.LINE_AA)
        else:
            rect, area = self.get_focus_object(frame)

            if rect:
                x, y, rw, rh = rect
                if area < self.min_scan_area:
                    status_msg = "MOVE CLOSER"
                    box_color  = (0, 255, 255)
                    self.buffer.append(2)
                    cv2.rectangle(frame, (x, y), (x + rw, y + rh), box_color, 2)
                else:
                    pad = 40
                    y1  = max(0, y - pad);   y2 = min(frame.shape[0], y + rh + pad)
                    x1  = max(0, x - pad);   x2 = min(frame.shape[1], x + rw + pad)
                    roi = frame[y1:y2, x1:x2]

                    if roi.size > 0:
                        # Pre-process image (replaces deprecated tf.keras.preprocessing)
                        img       = cv2.resize(roi, (224, 224))
                        img_rgb   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img_arr   = img_rgb.astype(np.float32) / 255.0
                        img_batch = np.expand_dims(img_arr, axis=0)

                        preds        = self.predict_image(img_batch)
                        current_pred = int(np.argmax(preds))
                        confidence   = float(preds[current_pred])
                        self.buffer.append(current_pred)

                        bad  = self.buffer.count(0)
                        good = self.buffer.count(1)

                        if bad >= self.defect_trigger:
                            status_msg = "DEFECT DETECTED"
                            box_color  = (0, 0, 255)
                            now = time.time()
                            if (now - self.last_log_time) > self.log_cooldown:
                                threading.Thread(
                                    target=self.save_log_to_db,
                                    args=(confidence,),
                                    daemon=True
                                ).start()
                                self.last_log_time = now
                        elif good > (len(self.buffer) // 2):
                            status_msg = "QUALITY PASS"
                            box_color  = (0, 255, 0)
                        else:
                            status_msg = "ANALYZING..."
                            box_color  = (0, 165, 255)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
            else:
                self.buffer.append(2)

            # Status banner
            cv2.rectangle(frame, (0, 0), (w, 55), box_color, -1)
            cv2.putText(frame, status_msg, (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        # ── Encode & return ──
        ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return jpeg.tobytes() if ret else _make_offline_frame()