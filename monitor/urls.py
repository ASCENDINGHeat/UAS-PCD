from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('feed/blended/', views.video_feed_blended, name='video_feed_blended'),
    path('feed/grayscale/', views.video_feed_grayscale, name='video_feed_grayscale'),
    path('feed/denoised/', views.video_feed_denoised, name='video_feed_denoised'),
    path('feed/thresholded/', views.video_feed_thresholded, name='video_feed_thresholded'),
    path('feed/morphological/', views.video_feed_morphological, name='video_feed_morphological'),
    path('change_source/', views.change_source, name='change_source'),
    path('toggle_stream/', views.toggle_stream, name='toggle_stream'),
    path('capture/', views.capture_still, name='capture_still'),
    path('upload/', views.upload_image, name='upload_image'),
    path('analyze/', views.analyze_captured_frame, name='analyze_captured_frame'),
]