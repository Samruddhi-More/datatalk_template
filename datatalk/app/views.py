from django.shortcuts import render
from rest_framework.generics import ListCreateAPIView, DestroyAPIView
from rest_framework.response import Response

from app.serializers import DataTalkSerializer
from app.models import DataTalkM
from app.datatalk import DataTalk

# Create your views here.

class ChatListCreateAV(ListCreateAPIView):

    serializer_class = DataTalkSerializer
    # permission_classes = [permissions.IsAuthenticated]
    # authentication_classes = [authentication.TokenAuthentication]
    queryset = DataTalkM.objects.all()

    def create(self, request, *args, **kwargs):

        serializer = DataTalkSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        instance = serializer.save()
        print(instance.file.url)

        chat = DataTalk(instance.file.url)
        # query = request.data.get('query')



        instance.response = chat.generate_response(instance.prompt)

        instance.save()

        return Response(serializer.data)


class DataTalkDeleteView(DestroyAPIView):
    serializer_class = DataTalkSerializer
    # permission_classes = [permissions.IsAuthenticated]
    # authentication_classes = [authentication.TokenAuthentication]
    queryset = DataTalkM.objects.all()




