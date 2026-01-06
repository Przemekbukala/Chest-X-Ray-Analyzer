from django.db import models
from django.contrib.auth.models import AbstractUser
from django.conf import settings

class CustomUser(AbstractUser):
    email = models.EmailField(unique=True, blank=True, null=True)


class XrayAnalysis(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="xray_analyses",
    )

    image = models.ImageField(
        upload_to="xrays/%Y/%m/%d/"
    )

    predicted_class = models.CharField(
        max_length=32
    )

    confidence = models.FloatField()

    probabilities = models.JSONField()

    heatmap = models.ImageField(
        upload_to="heatmaps/%Y/%m/%d/",
        null=True,
        blank=True,
    )

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user} - {self.predicted_class} ({self.created_at:%Y-%m-%d})"
