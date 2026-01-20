from django.test import TestCase, Client
from django.urls import reverse
from .models import CustomUser


class LoginTestCase(TestCase):
    def setUp(self):
        self.client = Client()
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


class RegisterTestCase(TestCase):
    def setUp(self):
        self.client = Client()
        self.register_url = reverse('register')
    
    def test_register_redirects_to_home(self):
        response = self.client.post(self.register_url, {
            'username': 'newuser',
            'email': 'newuser@example.com',
            'password1': 'SecurePass123!',
            'password2': 'SecurePass123!'
        })
        self.assertEqual(response.status_code, 302)
        self.assertRedirects(response, reverse('home'))
    
    def test_register_auto_login(self):
        self.client.post(self.register_url, {
            'username': 'autoLoginUser',
            'email': 'autologin@example.com',
            'password1': 'SecurePass123!',
            'password2': 'SecurePass123!'
        })
        upload_url = reverse('upload_xray')
        response = self.client.get(upload_url)
        self.assertEqual(response.status_code, 200)
    
    def test_register_duplicate_email_rejected(self):
        CustomUser.objects.create_user(
            username='user1',
            email='taken@example.com',
            password='Password123!'
        )
        response = self.client.post(self.register_url, {
            'username': 'newuser',
            'email': 'taken@example.com',
            'password1': 'SecurePass123!',
            'password2': 'SecurePass123!'
        })
        self.assertEqual(response.status_code, 200)
        self.assertFalse(CustomUser.objects.filter(username='newuser').exists())
    
    def test_register_without_email_allowed(self):
        response = self.client.post(self.register_url, {
            'username': 'noemailuser',
            'email': '',
            'password1': 'SecurePass123!',
            'password2': 'SecurePass123!'
        })
        self.assertEqual(response.status_code, 302)
        self.assertTrue(CustomUser.objects.filter(username='noemailuser').exists())
    
    def test_register_saves_email_correctly(self):
        self.client.post(self.register_url, {
            'username': 'emailtest',
            'email': 'specific@example.com',
            'password1': 'SecurePass123!',
            'password2': 'SecurePass123!'
        })
        user = CustomUser.objects.get(username='emailtest')
        self.assertEqual(user.email, 'specific@example.com')
        user = CustomUser.objects.get(username='emailtest')
        self.assertEqual(user.email, 'specific@example.com')

