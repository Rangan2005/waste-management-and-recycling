# accounts/urls.py
from django.urls import path
from .views import GarbageClassificationView
from rest_framework.authtoken.views import obtain_auth_token
from nirmal_bhoomi.views import ai_intro, ai, index, login, register

urlpatterns = [
    #path('register/', RegisterAPI.as_view(), name='register'),
    #path('login/', obtain_auth_token, name='login'),
    path('api/classify/', GarbageClassificationView.as_view(), name='classify_garbage'),
    path('ai', ai_intro, name='ai'),
    path('ai1', ai, name='ai1'),
    path('', index, name='home'),
    path('login', login, name='login'),
    path('register', register, name='register'),
]
