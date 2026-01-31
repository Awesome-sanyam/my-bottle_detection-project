import cv2
import os
import numpy as np
import tensorflow as tf
from django.conf import settings
from collections import deque
import threading
import time
import os
import sys
from .models import DefectLog, SystemConfig

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
                cls._instance.buffer = deque(maxlen=15)
            return cls._instance

    def initialize(self):
        """Lazy loader for camera hardware."""
        if self.initialized:
            return

        print("🎥 CAMERA: Opening Hardware...")
        self.video = cv2.VideoCapture(0)
        
        # High Performance Settings
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.video.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Load Config & Model
        self.load_configuration()
        
        self.last_log_time = 0
        self.log_cooldown = 5.0
        self.initialized = True

    def load_configuration(self):
        try:
            config = SystemConfig.objects.first()
            if config:
                print("⚙️ LOADING DB CONFIG...")
                self.min_scan_area = config.min_scan_area
                self.defect_trigger = config.defect_trigger_count
                
                if config.active_model:
                    model_path = config.active_model.path
                else:
                    model_path = os.path.join(settings.BASE_DIR, 'ml_model', 'bottle_model_multiclass.keras')
            else:
                print("⚠️ USING DEFAULTS.")
                self.min_scan_area = 45000
                self.defect_trigger = 4
                model_path = os.path.join(settings.BASE_DIR, 'ml_model', 'bottle_model_multiclass.keras')
        except Exception:
            # Fallback if DB is not ready yet
            self.min_scan_area = 45000
            self.defect_trigger = 4
            model_path = os.path.join(settings.BASE_DIR, 'ml_model', 'bottle_model_multiclass.keras')

        try:
            self.model = tf.keras.models.load_model(model_path)
            print("✅ MODEL LOADED")
        except Exception as e:
            print(f"❌ MODEL ERROR: {e}")
            self.model = None

    def hard_exit_handler(self, signum, frame):
        """Called by apps.py on CTRL+C"""
        print(f"\n☢️ SHUTDOWN SIGNAL RECEIVED. KILLING CAMERA...")
        if self.video and self.video.isOpened():
            self.video.release()
        os._exit(0)

    def save_log_to_db(self, confidence):
        try:
            DefectLog.objects.create(confidence=confidence)
            print(f"📝 LOG SAVED: {confidence:.2f}")
        except Exception:
            pass

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
                
                if roi.size > 0 and self.model:
                    img = cv2.resize(roi, (224, 224))
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0) / 255.0
                    
                    preds = self.model.predict(img_array, verbose=0)[0]
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

        cv2.rectangle(frame, (0, 0), (1280, 80), box_color, -1)
        cv2.putText(frame, status_msg, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()