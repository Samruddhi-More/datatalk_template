# Generated by Django 5.0.6 on 2024-07-23 10:20

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0003_alter_datatalkm_file_alter_datatalkm_response'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='datatalkm',
            name='file',
        ),
    ]
