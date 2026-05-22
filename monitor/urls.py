from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('feed/blended/', views.video_feed_blended, name='video_feed_blended'),
    path('change_source/', views.change_source, name='change_source'),
    path('toggle_stream/', views.toggle_stream, name='toggle_stream'),
    path('capture/', views.capture_still, name='capture_still'),
    path('analyze/', views.analyze_captured_frame, name='analyze_captured_frame'),
]