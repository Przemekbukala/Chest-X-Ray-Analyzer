from django.urls import path,include
from . import views


urlpatterns = [
    path('', views.home, name='home'),
    path('about/',views.about, name='about'),
    path('register/', views.register_view, name='register'), # will be/xray_analyzer_app/register/ instead of /account/register
    path('upload/', views.upload_xray, name='upload_xray'),
    path("history/", views.xray_history, name="xray_history"),
]