# accounts/urls.py
from django.urls import path
from .views import GarbageClassificationView
from nirmal_bhoomi.views import ai_intro, ai, index, login, register, index1, submit_contact

urlpatterns = [
    path('api/classify/', GarbageClassificationView.as_view(), name='classify_garbage'),
    path('ai', ai_intro, name='ai'),
    path('ai1', ai, name='ai1'),
    path('', index, name='home'),
    path('login', login, name='login'),
    path('register', register, name='register'),
    path('index1', index1, name='home_aft_login'),
    path('submit_contact/', submit_contact, name='submit_contact'),
]
