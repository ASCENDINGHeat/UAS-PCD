from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('feed/blended/', views.video_feed_blended, name='video_feed_blended'),
    path('change_source/', views.change_source, name='change_source'),
]