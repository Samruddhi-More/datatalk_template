from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static


from app.views import *

urlpatterns = [
    path('chat', ChatListCreateAV.as_view(), name='chat'),
    path('chat/<int:pk>', DataTalkDeleteView.as_view(), name='chat-delete')
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)