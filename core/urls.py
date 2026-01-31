from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    # Route root URL '' to the detector app
    path('', include('detector.urls')),
]