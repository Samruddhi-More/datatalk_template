from rest_framework import serializers

from app.models import *

class DataTalkSerializer(serializers.ModelSerializer):
    class Meta:
        model = DataTalkM
        fields = "__all__"