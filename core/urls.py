from django.contrib import admin
from django.urls import path, include
from detector.views import live_monitor_admin

urlpatterns = [
    path('admin/live-monitor/', live_monitor_admin, name='admin_live_monitor'),
    path('admin/', admin.site.urls),
    # Route root URL '' to the detector app
    path('', include('detector.urls')),
]