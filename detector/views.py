from django.shortcuts import render, redirect
from django.http import StreamingHttpResponse, HttpResponse
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login
from .camera import VideoCamera
from .models import DefectLog
import csv
from django.utils import timezone
from datetime import timedelta

# Singleton Access
camera_instance = VideoCamera()

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
    recent_logs = DefectLog.objects.order_by('-timestamp')[:5]
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
    # Camera init is handled by apps.py or lazy load
    # We just grab frames here
    while True:
        frame = camera.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def video_feed(request):
    return StreamingHttpResponse(gen(camera_instance),
                                 content_type='multipart/x-mixed-replace; boundary=frame')