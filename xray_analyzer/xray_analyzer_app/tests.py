from django.test import TestCase, Client
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from .models import CustomUser, XrayAnalysis
from unittest.mock import patch, MagicMock
import io


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


class UploadXrayTestCase(TestCase):
    def setUp(self):
        self.client = Client()
        self.upload_url = reverse('upload_xray')
        self.test_password = 'TestPassword123!'
        self.test_user = CustomUser.objects.create_user(
            username='testuser',
            email='test@example.com',
            password=self.test_password
        )
        self.client.login(username='testuser', password=self.test_password)
    
    def test_upload_no_file_rejected(self):
        response = self.client.post(self.upload_url, {})
        self.assertEqual(response.status_code, 302)
        self.assertRedirects(response, self.upload_url)
    
    def test_upload_png_rejected(self):
        png_file = SimpleUploadedFile(
            name='test.png',
            content=b'fake png content',
            content_type='image/png'
        )
        response = self.client.post(self.upload_url, {'xray_file': png_file})
        self.assertEqual(response.status_code, 302)
        self.assertRedirects(response, self.upload_url)
        self.assertEqual(XrayAnalysis.objects.count(), 0)
    
    def test_upload_pdf_rejected(self):
        pdf_file = SimpleUploadedFile(
            name='test.pdf',
            content=b'fake pdf content',
            content_type='application/pdf'
        )
        response = self.client.post(self.upload_url, {'xray_file': pdf_file})
        self.assertEqual(response.status_code, 302)
        self.assertRedirects(response, self.upload_url)
        self.assertEqual(XrayAnalysis.objects.count(), 0)
    
    def test_upload_file_too_large_rejected(self):
        large_content = b'x' * (11 * 1024 * 1024)
        large_file = SimpleUploadedFile(
            name='large.jpg',
            content=large_content,
            content_type='image/jpeg'
        )
        response = self.client.post(self.upload_url, {'xray_file': large_file})
        self.assertEqual(response.status_code, 302)
        self.assertRedirects(response, self.upload_url)
        self.assertEqual(XrayAnalysis.objects.count(), 0)
    
    @patch('xray_analyzer_app.views.ImageAnalizer')
    @patch('xray_analyzer_app.views.get_device')
    def test_upload_valid_jpg_creates_analysis(self, mock_get_device, mock_analyzer_class):
        mock_get_device.return_value = 'cpu'
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = {'normal': 85.5, 'pneumonia': 10.0, 'tuberculosis': 4.5}
        mock_analyzer_class.return_value = mock_analyzer
        
        jpg_file = SimpleUploadedFile(
            name='xray.jpg',
            content=b'\xff\xd8\xff\xe0' + b'fake jpg content',
            content_type='image/jpeg'
        )
        response = self.client.post(self.upload_url, {'xray_file': jpg_file})
        
        self.assertEqual(XrayAnalysis.objects.count(), 1)
        analysis = XrayAnalysis.objects.first()
        self.assertEqual(analysis.user, self.test_user)
    
    @patch('xray_analyzer_app.views.ImageAnalizer')
    @patch('xray_analyzer_app.views.get_device')
    def test_upload_valid_jpeg_creates_analysis(self, mock_get_device, mock_analyzer_class):
        mock_get_device.return_value = 'cpu'
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = {'normal': 85.5, 'pneumonia': 10.0, 'tuberculosis': 4.5}
        mock_analyzer_class.return_value = mock_analyzer
        
        jpeg_file = SimpleUploadedFile(
            name='xray.jpeg',
            content=b'\xff\xd8\xff\xe0' + b'fake jpeg content',
            content_type='image/jpeg'
        )
        response = self.client.post(self.upload_url, {'xray_file': jpeg_file})
        
        self.assertEqual(XrayAnalysis.objects.count(), 1)
    
    @patch('xray_analyzer_app.views.ImageAnalizer')
    @patch('xray_analyzer_app.views.get_device')
    def test_upload_redirects_to_analysis_details(self, mock_get_device, mock_analyzer_class):
        mock_get_device.return_value = 'cpu'
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = {'normal': 85.5, 'pneumonia': 10.0, 'tuberculosis': 4.5}
        mock_analyzer_class.return_value = mock_analyzer
        
        jpg_file = SimpleUploadedFile(
            name='xray.jpg',
            content=b'\xff\xd8\xff\xe0' + b'fake jpg content',
            content_type='image/jpeg'
        )
        response = self.client.post(self.upload_url, {'xray_file': jpg_file})
        
        analysis = XrayAnalysis.objects.first()
        self.assertEqual(response.status_code, 302)
        self.assertIn(f'/details/{analysis.pk}/', response.url)
    
    @patch('xray_analyzer_app.views.ImageAnalizer')
    @patch('xray_analyzer_app.views.get_device')
    def test_upload_analysis_error_sets_error_status(self, mock_get_device, mock_analyzer_class):
        mock_get_device.return_value = 'cpu'
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.side_effect = Exception('Model not found')
        mock_analyzer_class.return_value = mock_analyzer
        
        jpg_file = SimpleUploadedFile(
            name='xray.jpg',
            content=b'\xff\xd8\xff\xe0' + b'fake jpg content',
            content_type='image/jpeg'
        )
        response = self.client.post(self.upload_url, {'xray_file': jpg_file})
        
        analysis = XrayAnalysis.objects.first()
        self.assertEqual(analysis.predicted_class, 'error')
    
    @patch('xray_analyzer_app.views.ImageAnalizer')
    @patch('xray_analyzer_app.views.get_device')
    def test_upload_analysis_saves_probabilities(self, mock_get_device, mock_analyzer_class):
        mock_get_device.return_value = 'cpu'
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = {'normal': 70.0, 'pneumonia': 20.0, 'tuberculosis': 10.0}
        mock_analyzer_class.return_value = mock_analyzer
        
        jpg_file = SimpleUploadedFile(
            name='xray.jpg',
            content=b'\xff\xd8\xff\xe0' + b'fake jpg content',
            content_type='image/jpeg'
        )
        self.client.post(self.upload_url, {'xray_file': jpg_file})
        
        analysis = XrayAnalysis.objects.first()
        self.assertEqual(analysis.predicted_class, 'normal')
        self.assertEqual(analysis.confidence, 70.0)
        self.assertEqual(analysis.probabilities['pneumonia'], 20.0)
