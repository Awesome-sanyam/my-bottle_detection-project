from django.db import models
from django.utils import timezone
from django.core.exceptions import ValidationError

class DefectLog(models.Model):
    timestamp = models.DateTimeField(default=timezone.now)
    event_type = models.CharField(max_length=50, default="DEFECT_DETECTED")
    confidence = models.FloatField(help_text="AI Confidence Score (0-1)")

    def __str__(self):
        return f"Defect at {self.timestamp.strftime('%H:%M:%S')} ({self.confidence:.2f})"

class SystemConfig(models.Model):
    site_name = models.CharField(max_length=100, default="QC-Vision Main")
    active_model = models.FileField(upload_to='ml_models/', blank=True, null=True, help_text="Upload custom .keras model")
    
    # Tuning Parameters
    confidence_threshold = models.FloatField(default=0.60)
    defect_trigger_count = models.IntegerField(default=4, help_text="Consecutive frames to trigger alert")
    min_scan_area = models.IntegerField(default=45000, help_text="Minimum pixel size to scan")

    def save(self, *args, **kwargs):
        if not self.pk and SystemConfig.objects.exists():
            raise ValidationError('There can be only one SystemConfig instance')
        return super(SystemConfig, self).save(*args, **kwargs)

    def __str__(self):
        return "System Configuration"