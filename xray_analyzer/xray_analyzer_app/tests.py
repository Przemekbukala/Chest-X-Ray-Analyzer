from django.test import TestCase, Client
from django.urls import reverse
from .models import CustomUser


class LoginTestCase(TestCase):
    def setUp(self):
        self.client = Client()
        self.login_url = reverse('login')
        
        self.test_password = 'TestPassword123!'
        self.test_user = CustomUser.objects.create_user(
            username='testuser',
            email='test@example.com',
            password=self.test_password
        )
    
    def test_authenticated_user_access(self):
        self.client.login(username='testuser', password=self.test_password)
        
        upload_url = reverse('upload_xray')
        response = self.client.get(upload_url)
        self.assertEqual(response.status_code, 200)
    
    def test_unauthenticated_user_redirect(self):
        upload_url = reverse('upload_xray')
        response = self.client.get(upload_url)
        self.assertEqual(response.status_code, 302)
        self.assertIn('login', response.url)
