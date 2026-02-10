import cv2
import os
import numpy as np
import threading
import time
import os
import sys
from collections import deque
from django.conf import settings
from .models import DefectLog, SystemConfig

# --- HYBRID IMPORT LOGIC ---
# Try to import full TensorFlow (Mac). If that fails, import TFLite Runtime (Pi).
try:
    import tensorflow as tf
    USING_TFLITE = False
    print("🖥️ PLATFORM DETECTED: HIGH-POWER SERVER (Mac/PC)")
except ImportError:
    import tflite_runtime.interpreter as tflite
    USING_TFLITE = True
    print("⚡ PLATFORM DETECTED: EDGE DEVICE (Raspberry Pi)")

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
                cls._instance.interpreter = None # For TFLite
                cls._instance.buffer = deque(maxlen=15)
            return cls._instance

    def initialize(self):
        if self.initialized: return

        print("🎥 CAMERA: Opening Hardware...")
        self.video = cv2.VideoCapture(0)
        
        # Pi Camera Optimization (Low Resolution for Speed)
        if USING_TFLITE:
            self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        else:
            self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.load_configuration()
        self.last_log_time = 0
        self.log_cooldown = 5.0
        self.initialized = True

    def load_configuration(self):
        # ... (Config loading logic same as before, skipping for brevity) ...
        # Default fallback logic:
        self.min_scan_area = 45000
        self.defect_trigger = 4
        
        # MODEL LOADING
        if USING_TFLITE:
            model_path = os.path.join(settings.BASE_DIR, 'ml_model', 'bottle_model_quantized.tflite')
            print(f"⚡ LOADING TFLITE MODEL: {model_path}")
            try:
                self.interpreter = tflite.Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                
                # Get input/output details
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print("✅ TFLITE ENGINE READY")
            except Exception as e:
                print(f"❌ TFLITE LOAD ERROR: {e}")
        else:
            # Standard Keras Load (Mac)
            model_path = os.path.join(settings.BASE_DIR, 'ml_model', 'bottle_model_multiclass.keras')
            print(f"🏭 LOADING KERAS MODEL: {model_path}")
            try:
                self.model = tf.keras.models.load_model(model_path)
                print("✅ KERAS ENGINE READY")
            except Exception as e:
                print(f"❌ KERAS LOAD ERROR: {e}")

    def predict_image(self, img_array):
        """Unified Prediction Method"""
        if USING_TFLITE and self.interpreter:
            # TFLite Inference
            # Ensure input is float32
            input_data = np.array(img_array, dtype=np.float32)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            preds = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            return preds
            
        elif not USING_TFLITE and self.model:
            # Keras Inference
            return self.model.predict(img_array, verbose=0)[0]
        return [0, 0, 1] # Default to Other if fail

    # ... (Keep release_resources, hard_exit_handler, save_log_to_db, get_focus_object exactly as before) ...

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
                
                if roi.size > 0:
                    img = cv2.resize(roi, (224, 224))
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0) / 255.0
                    
                    # CALL THE UNIFIED PREDICTION METHOD
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

        cv2.rectangle(frame, (0, 0), (1280 if not USING_TFLITE else 640, 60), box_color, -1)
        cv2.putText(frame, status_msg, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()