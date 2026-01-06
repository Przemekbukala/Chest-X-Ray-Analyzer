from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm
from .forms import CustomUserCreationForm
from .models import XrayAnalysis
from .image_analyzer import ImageAnalizer
from . import model_loader
import logging
logger = logging.getLogger(__name__)

def home(request):
    return render(request, 'home.html')

def about(request):
    return render(request, 'about.html')

# def login_view(request):
#     if request.method == "POST":
#         username = request.POST.get("username")
#         password = request.POST.get("password")
#
#         user = authenticate(request, username=username, password=password)
#
#         if user is not None:
#             login(request, user)
#             return redirect('home')
#         else:
#             return HttpResponse("Invalid username or password")
#
#     return render(request, 'registration/login.html')

def register_view(request):
    if request.method == "POST":
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            login(request,form.save())
            return redirect('home')
    else:
        form = CustomUserCreationForm()

    return render(request, "registration/register.html", {"form": form})

import os
from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings
from django.contrib.auth.decorators import login_required

@login_required
def upload_xray(request):
    if request.method == 'POST':
        if 'xray_file' not in request.FILES:
            messages.error(request, 'No file was chosen')
            return redirect('upload_xray')

        file = request.FILES['xray_file']

        if not file.name.lower().endswith(('.jpg')):
            messages.error(request, 'Invalid file format, JPG required')
            return redirect('upload_xray')

        if file.size > 10 * 1024 * 1024:
            messages.error(request, 'File is too big, exceeds 10 MB')
            return redirect('upload_xray')

        analysis = XrayAnalysis.objects.create(
            user=request.user,
            image=file,
            predicted_class="pending",
            confidence=0.0,
            probabilities={},
        )
        messages.success(request, f'File "{file.name}" successfully saved')

        try:
            analyzer = ImageAnalizer()
            result = analyzer.analyze(analysis.image.path)

            analysis.predicted_class = result["predicted_class"]
            analysis.confidence = result["confidence"]
            analysis.probabilities = result["probabilities"]
            analysis.save()

            messages.info(request, f'Analysis result:\n Normal {result["probabilities"]["normal"]}%\n Pneumonia {result["probabilities"]['pneumonia']}%\n Tuberculosis {result["probabilities"]['tuberculosis']}%')

        except Exception as e:
            analysis.predicted_class = "error"
            analysis.save()

            logger.exception("X-ray analysis failed")
            messages.error(request, "Analysis failed. Please try again.")


        return redirect('upload_xray')

    return render(request, 'upload_xray.html')

@login_required
def xray_history(request):
    analyses = (
        XrayAnalysis.objects
        .filter(user=request.user)
        .order_by("-created_at")
    )

    return render(
        request,
        "xray_history.html",
        {
            "analyses": analyses
        }
    )