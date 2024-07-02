from django.urls import path, include

from app.views import Chat

urlpatterns = [
    path('chat', Chat.as_view(), name='chat')
]
