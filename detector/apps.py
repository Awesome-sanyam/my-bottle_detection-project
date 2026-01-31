from django.apps import AppConfig
import sys
import signal
import os

class DetectorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'detector'

    def ready(self):
        """
        This runs ONCE at startup in the Main Thread.
        We initialize the camera here to avoid threading errors.
        """
        # Only run in the main process (prevents double loading)
        if 'runserver' in sys.argv and os.environ.get('RUN_MAIN') != 'true':
            return
            
        from .camera import VideoCamera
        
        print("🚀 SYSTEM STARTUP: Initializing Industrial Camera...")
        camera = VideoCamera()
        
        # Register the Hard Exit signals here (Main Thread Only)
        try:
            signal.signal(signal.SIGINT, camera.hard_exit_handler)
            signal.signal(signal.SIGTERM, camera.hard_exit_handler)
            print("✅ SIGNAL HANDLERS REGISTERED")
        except ValueError:
            print("⚠️ WARNING: Signal handlers could not be registered (Thread Issue).")