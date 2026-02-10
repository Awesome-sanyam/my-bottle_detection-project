from django.contrib import admin
from .models import DefectLog, SystemConfig

@admin.register(DefectLog)
class DefectLogAdmin(admin.ModelAdmin):
    list_display = ('timestamp', 'event_type', 'confidence')
    list_filter = ('event_type', 'timestamp')
    date_hierarchy = 'timestamp'
    ordering = ('-timestamp',)

@admin.register(SystemConfig)
class SystemConfigAdmin(admin.ModelAdmin):
    # Only allow one config to exist
    def has_add_permission(self, request):
        return not SystemConfig.objects.exists()