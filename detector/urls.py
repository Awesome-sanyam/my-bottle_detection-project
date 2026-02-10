from django.urls import path, include
from . import views

urlpatterns = [
    # Dashboard (Protected)
    path('', views.index, name='index'),
    
    # Auth System
    path('accounts/', include('django.contrib.auth.urls')), # Provides login/logout logic automatically
    path('register/', views.register, name='register'),
    
    # Features
    path('video_feed', views.video_feed, name='video_feed'),
    path('download_csv/', views.download_logs, name='download_logs'),
]