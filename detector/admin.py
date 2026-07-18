from django.contrib import admin
from django.utils.html import format_html
from django.urls import path
from django.http import HttpResponse
from django.template.response import TemplateResponse
from .models import DefectLog, SystemConfig
import platform


# ── Custom Admin Site ────────────────────────────────────────────────────────
class QCVisionAdminSite(admin.AdminSite):
    site_header  = "QC-Vision Pro  |  Admin Control"
    site_title   = "QC-Vision Admin"
    index_title  = "System Control Panel"

    def get_urls(self):
        urls = super().get_urls()
        custom = [
            path('live-monitor/', self.admin_view(self.live_monitor_view),
                 name='live_monitor'),
        ]
        return custom + urls

    def live_monitor_view(self, request):
        """Full-screen live camera view accessible from the admin panel."""
        from .models import DefectLog
        recent_logs = DefectLog.objects.order_by('-timestamp')[:20]
        context = {
            **self.each_context(request),
            'title': 'Live Camera Monitor',
            'logs':  recent_logs,
            'platform': platform.system(),
        }
        return TemplateResponse(request, 'admin/live_monitor.html', context)


admin_site = QCVisionAdminSite(name='qcadmin')


# ── DefectLog Admin ──────────────────────────────────────────────────────────
@admin.register(DefectLog, site=admin_site)
class DefectLogAdmin(admin.ModelAdmin):
    list_display    = ('timestamp', 'event_type', 'confidence_bar', 'confidence')
    list_filter     = ('event_type',)
    date_hierarchy  = 'timestamp'
    ordering        = ('-timestamp',)
    readonly_fields = ('timestamp',)

    def confidence_bar(self, obj):
        pct   = int(obj.confidence * 100)
        color = '#ef4444' if pct > 70 else '#f59e0b' if pct > 40 else '#10b981'
        return format_html(
            '<div style="width:120px;background:#1f2937;border-radius:4px;overflow:hidden;">'
            '<div style="width:{}%;background:{};height:16px;border-radius:4px;"></div>'
            '</div>',
            pct, color
        )
    confidence_bar.short_description = 'Confidence'


# ── SystemConfig Admin ───────────────────────────────────────────────────────
@admin.register(SystemConfig, site=admin_site)
class SystemConfigAdmin(admin.ModelAdmin):
    fieldsets = (
        ('General', {
            'fields': ('site_name', 'active_model')
        }),
        ('Detection Tuning', {
            'fields': ('confidence_threshold', 'defect_trigger_count', 'min_scan_area'),
            'description': 'Adjust these to tune sensitivity vs. false-alarm rate.'
        }),
    )

    def has_add_permission(self, request):
        return not SystemConfig.objects.exists()

    def has_delete_permission(self, request, obj=None):
        return False     # prevent accidental deletion of the single config


# ── Also register on default admin (for /admin/ URL) ────────────────────────
@admin.register(DefectLog)
class DefectLogDefaultAdmin(admin.ModelAdmin):
    list_display    = ('timestamp', 'event_type', 'confidence_bar', 'confidence')
    list_filter     = ('event_type',)
    date_hierarchy  = 'timestamp'
    ordering        = ('-timestamp',)
    readonly_fields = ('timestamp',)

    def confidence_bar(self, obj):
        pct   = int(obj.confidence * 100)
        color = '#ef4444' if pct > 70 else '#f59e0b' if pct > 40 else '#10b981'
        return format_html(
            '<div style="width:120px;background:#1f2937;border-radius:4px;overflow:hidden;">'
            '<div style="width:{}%;background:{};height:16px;border-radius:4px;"></div>'
            '</div>',
            pct, color
        )
    confidence_bar.short_description = 'Confidence'


@admin.register(SystemConfig)
class SystemConfigDefaultAdmin(admin.ModelAdmin):
    fieldsets = (
        ('General', {
            'fields': ('site_name', 'active_model')
        }),
        ('Detection Tuning', {
            'fields': ('confidence_threshold', 'defect_trigger_count', 'min_scan_area'),
            'description': 'Adjust these values to tune detection sensitivity.'
        }),
    )

    def has_add_permission(self, request):
        return not SystemConfig.objects.exists()

    def has_delete_permission(self, request, obj=None):
        return False