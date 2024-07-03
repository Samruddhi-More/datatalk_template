from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class DataTalkM(models.Model):
    file = models.FileField(upload_to='uploads/', null=True, blank=True)
    prompt = models.CharField(max_length=255)
    response = models.TextField(null=True, blank=True)
    updated_at = models.DateField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)

