from django.apps import AppConfig


class XrayAnalyzerAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'xray_analyzer_app'

    def ready(self):
        from .model_loader import load_model
        #load_model()