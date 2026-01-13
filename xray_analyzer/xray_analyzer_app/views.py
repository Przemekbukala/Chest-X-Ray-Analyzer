from django.http import HttpResponse
from django.shortcuts import get_object_or_404, render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm
from .forms import CustomUserCreationForm
from .models import XrayAnalysis
from .image_analyzer import ImageAnalizer
from . import model_loader
import logging
import time
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

        if not file.name.lower().endswith(('.jpg', '.jpeg')):
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

        try:
            analyzer = ImageAnalizer()
            
            start_time = time.time()
            
            results = analyzer.analyze(analysis.image.path)

            if results:
                best_class = max(results, key=results.get)
                
                heatmap_filename = f"heatmap_{analysis.id}.jpg"
                heatmap_relative_path = os.path.join('heatmaps', heatmap_filename)
                heatmap_full_path = os.path.join(settings.MEDIA_ROOT, heatmap_relative_path)
                
                os.makedirs(os.path.dirname(heatmap_full_path), exist_ok=True)
                
                analyzer.compute_heatmap(heatmap_full_path)
                
                duration = time.time() - start_time
                logger.info(f"Analysis completed in: {duration:.2f}s")

                analysis.predicted_class = best_class
                analysis.confidence = results[best_class]
                analysis.probabilities = results
                analysis.heatmap = heatmap_relative_path
                analysis.save()

                messages.success(request, f'Analysis successful ({duration:.2f}s)')
                
                msg = f"Result: {best_class.upper()} ({results[best_class]}%)"
                messages.info(request, msg)
            
        except Exception as e:
            analysis.predicted_class = "error"
            analysis.save()
            logger.exception(f"X-ray analysis failed: {e}")
            messages.error(request, "Analysis failed. Please check if model.pth exists.")

        return redirect('xray_history') 

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

@login_required
def delete_analysis(request, pk):
    analysis = get_object_or_404(XrayAnalysis, pk=pk, user=request.user)
    
    if analysis.image:
        analysis.image.delete(save=False)
    
    if hasattr(analysis, 'heatmap') and analysis.heatmap:
        analysis.heatmap.delete(save=False)
        
    analysis.delete()
    return redirect('xray_history')

@login_required
def analysis_details(request, pk):
    analysis = get_object_or_404(XrayAnalysis, pk=pk, user=request.user)

    probs = analysis.probabilities or {}
    p_normal = float(probs.get("normal", 0.0) or 0.0)
    p_pneumonia = float(probs.get("pneumonia", 0.0) or 0.0)
    p_tuberculosis = float(probs.get("tuberculosis", 0.0) or 0.0)

    survival_prob = round(p_normal, 2)

    predicted = (analysis.predicted_class or "").lower()
    if predicted == "pneumonia":
        second_disease_name = "tuberculosis"
        second_disease_prob = round(p_tuberculosis, 2)
    elif predicted == "tuberculosis":
        second_disease_name = "pneumonia"
        second_disease_prob = round(p_pneumonia, 2)
    else:
        if p_pneumonia >= p_tuberculosis:
            second_disease_name = "pneumonia"
            second_disease_prob = round(p_pneumonia, 2)
        else:
            second_disease_name = "tuberculosis"
            second_disease_prob = round(p_tuberculosis, 2)

    analyses = (
        XrayAnalysis.objects
        .filter(user=request.user)
        .order_by("-created_at")
    )

    context = {
        "analyses": analyses,
        "analysis": analysis,
        "survival_prob": survival_prob,
        "second_disease_name": second_disease_name,
        "second_disease_prob": second_disease_prob,
        "selected_id": analysis.pk,
        "show_details": True,
    }
    return render(request, "xray_history.html", context)

