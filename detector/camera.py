import cv2
import os
import numpy as np
import threading
import time
import sys
from collections import deque
from django.conf import settings
from .models import DefectLog, SystemConfig

# --- INTELLIGENT IMPORT LOGIC ---
# 1. Check which model file exists on the disk
TFLITE_MODEL_PATH = os.path.join(settings.BASE_DIR, 'ml_model', 'bottle_model_quantized.tflite')
USE_TFLITE_MODE = os.path.exists(TFLITE_MODEL_PATH)

try:
    import tensorflow as tf
    if USE_TFLITE_MODE:
        # Use the Interpreter from full TensorFlow
        Interpreter = tf.lite.Interpreter
        print(f"⚡ PLATFORM: EDGE DEVICE (Using TensorFlow Lite via Full TF)")
    else:
        print(f"🖥️ PLATFORM: HIGH-POWER SERVER (Using Keras)")
except ImportError:
    # Fallback for devices with only runtime
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
    print(f"⚡ PLATFORM: EDGE DEVICE (Using TensorFlow Lite Runtime)")

class VideoCamera(object):
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(VideoCamera, cls).__new__(cls)
                cls._instance.initialized = False
                cls._instance.video = None
                cls._instance.model = None
                cls._instance.interpreter = None
                cls._instance.buffer = deque(maxlen=15)
            return cls._instance

    def initialize(self):
        if self.initialized: return

        print("🎥 CAMERA: Opening Hardware (V4L2 + MJPEG Force)...")
        
        # --- CRITICAL PI FIX: Force V4L2 Driver ---
        self.video = cv2.VideoCapture(0, cv2.CAP_V4L2)
        
        # --- CRITICAL PI FIX: Force MJPEG Format ---
        # This prevents the "Monster Image" / Raw Data crash
        self.video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        # Set Standard Resolution
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.load_configuration()
        self.last_log_time = 0
        self.log_cooldown = 5.0
        self.initialized = True

    def load_configuration(self):
        try:
            self.min_scan_area = 45000
            self.defect_trigger = 4
            
            # --- MODEL LOADING LOGIC ---
            if USE_TFLITE_MODE:
                print(f"⚡ LOADING TFLITE MODEL: {TFLITE_MODEL_PATH}")
                try:
                    self.interpreter = Interpreter(model_path=TFLITE_MODEL_PATH)
                    self.interpreter.allocate_tensors()
                    self.input_details = self.interpreter.get_input_details()
                    self.output_details = self.interpreter.get_output_details()
                    print("✅ TFLITE ENGINE READY")
                except Exception as e:
                    print(f"❌ TFLITE LOAD ERROR: {e}")
            else:
                keras_path = os.path.join(settings.BASE_DIR, 'ml_model', 'bottle_model_multiclass.keras')
                print(f"🏭 LOADING KERAS MODEL: {keras_path}")
                try:
                    self.model = tf.keras.models.load_model(keras_path)
                    print("✅ KERAS ENGINE READY")
                except Exception as e:
                    print(f"❌ KERAS LOAD ERROR: {e}")

        except Exception as e:
            print(f"❌ CONFIG ERROR: {e}")

    def predict_image(self, img_array):
        if USE_TFLITE_MODE and self.interpreter:
            input_data = np.array(img_array, dtype=np.float32)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            preds = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            return preds
        elif not USE_TFLITE_MODE and self.model:
            return self.model.predict(img_array, verbose=0)[0]
        return [0, 0, 1]

    def hard_exit_handler(self, signum, frame):
        print(f"\n☢️ SHUTDOWN SIGNAL RECEIVED...")
        if self.video and self.video.isOpened(): self.video.release()
        os._exit(0)

    def save_log_to_db(self, confidence):
        try:
            DefectLog.objects.create(confidence=confidence)
            print(f"📝 LOG SAVED: {confidence:.2f}")
        except: pass

    def get_focus_object(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blur, 50, 150)
        kernel = np.ones((5,5), np.uint8)
        dilated = cv2.dilate(edged, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None, 0
        valid_cnts = [c for c in contours if cv2.contourArea(c) > 5000]
        if not valid_cnts: return None, 0
        c = max(valid_cnts, key=cv2.contourArea)
        return cv2.boundingRect(c), cv2.contourArea(c)

    def get_frame(self):
        if not self.initialized: self.initialize()
        if not self.video or not self.video.isOpened(): return None

        success, frame = self.video.read()
        if not success: return None
        
        # --- SANITY CHECK (Prevents C++ Crash) ---
        h, w = frame.shape[:2]
        if w > 1000 or h > 1000: 
            return None # Skip garbage frame
            
        # --- FORCE COLOR (Prevents Tuple Crash) ---
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif len(frame.shape) == 3 and frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        rect, area = self.get_focus_object(frame)
        status_msg = "WAITING..."
        box_color = (100, 100, 100)

        if rect:
            x, y, w, h = rect
            if area < self.min_scan_area:
                status_msg = "⚠️ MOVE CLOSER"
                box_color = (0, 255, 255)
                self.buffer.append(2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
            else:
                pad = 40
                y1, y2 = max(0, y-pad), min(frame.shape[0], y+h+pad)
                x1, x2 = max(0, x-pad), min(frame.shape[1], x+w+pad)
                roi = frame[y1:y2, x1:x2]
                
                if roi.size > 0:
                    img = cv2.resize(roi, (224, 224))
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0) / 255.0
                    
                    preds = self.predict_image(img_array)
                    
                    current_pred = np.argmax(preds)
                    confidence = preds[current_pred]
                    self.buffer.append(current_pred)

                    bad = self.buffer.count(0)
                    good = self.buffer.count(1)

                    if bad >= self.defect_trigger:
                        status_msg = "🚫 DEFECT DETECTED"
                        box_color = (0, 0, 255)
                        current_time = time.time()
                        if (current_time - self.last_log_time) > self.log_cooldown:
                            threading.Thread(target=self.save_log_to_db, args=(confidence,)).start()
                            self.last_log_time = current_time
                    elif good > (len(self.buffer) // 2):
                        status_msg = "✅ QUALITY PASS"
                        box_color = (0, 255, 0)
                    else:
                        status_msg = "🔍 ANALYZING..."
                        box_color = (0, 165, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
        else:
            self.buffer.append(2)

        cv2.rectangle(frame, (0, 0), (640, 60), box_color, -1)
        cv2.putText(frame, status_msg, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # --- SAFE ENCODING ---
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret: return None
        return jpeg.tobytes()