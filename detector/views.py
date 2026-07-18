from django.shortcuts import render, redirect
from django.http import StreamingHttpResponse, HttpResponse
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth import login
from django.contrib.admin.views.decorators import staff_member_required
from .camera import VideoCamera
from .models import DefectLog
import csv
import platform
from django.utils import timezone
from datetime import timedelta

# ── Lazy singleton — only initialised on first request, not at import ──
_camera_instance = None

def get_camera():
    global _camera_instance
    if _camera_instance is None:
        _camera_instance = VideoCamera()
    return _camera_instance


def register(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('index')
    else:
        form = UserCreationForm()
    return render(request, 'detector/register.html', {'form': form})


@login_required
def index(request):
    recent_logs = DefectLog.objects.order_by('-timestamp')[:10]
    return render(request, 'detector/dashboard.html', {'logs': recent_logs})


@login_required
def download_logs(request):
    one_day_ago = timezone.now() - timedelta(days=1)
    logs = DefectLog.objects.filter(timestamp__gte=one_day_ago).order_by('-timestamp')

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="defects_report.csv"'

    writer = csv.writer(response)
    writer.writerow(['Timestamp', 'Type', 'Confidence'])
    for log in logs:
        writer.writerow([log.timestamp, log.event_type, log.confidence])

    return response


def gen(camera):
    """MJPEG frame generator. get_frame() always returns bytes now (never None)."""
    while True:
        frame = camera.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@login_required
def video_feed(request):
    return StreamingHttpResponse(
        gen(get_camera()),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )


@staff_member_required
def live_monitor_admin(request):
    """Admin-only live camera monitoring page."""
    recent_logs = DefectLog.objects.order_by('-timestamp')[:20]
    return render(request, 'admin/live_monitor.html', {
        'title':    'Live Camera Monitor',
        'logs':     recent_logs,
        'platform': platform.system(),
    })