# Generated by Django 5.0.6 on 2024-07-03 11:58

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0002_rename_datatalk_datatalkm'),
    ]

    operations = [
        migrations.AlterField(
            model_name='datatalkm',
            name='file',
            field=models.FileField(blank=True, null=True, upload_to='uploads/'),
        ),
        migrations.AlterField(
            model_name='datatalkm',
            name='response',
            field=models.TextField(blank=True, null=True),
        ),
    ]
