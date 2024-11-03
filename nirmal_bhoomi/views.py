from rest_framework import status
from rest_framework.response import Response
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login as auth_login
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from django.conf import settings
from django.contrib import messages
import os
from django.shortcuts import render, redirect
from ultralytics import YOLO
from django.http import JsonResponse
from .forms import ContactForm 

def ai_intro(request):
    return render(request, 'ai-intro.html')

def ai(request):
    return render(request, 'ai.html')

def index(request):
    return render(request, 'index.html')

def index1(request):
    return render(request, 'index_aft_log.html')

def login(request):
    if request.method == "POST": 
        username = request.POST['username']
        pass1 = request.POST['pass1']

        user = authenticate(request, username=username, password=pass1)

        if user is not None:
            auth_login(request, user)
            return render(request, "index_aft_log.html")
        else:
            messages.error(request, "Invalid Credentials")
            return redirect('login') 

    return render(request, 'LOGIN.html')

def register(request):
    if request.method == "POST":
        username = request.POST['username']
        email = request.POST['email']
        pass1 = request.POST['pass1']
        pass2 = request.POST['pass2']

        if pass1 != pass2:
            messages.error(request, "Passwords do not match")
            return redirect('register')
        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already taken. Please choose a different one.")
            return redirect('register')
        if User.objects.filter(email=email).exists():
            messages.error(request, "An account with this email already exists.")
            return redirect('register')

        myuser = User.objects.create_user(username, email, pass1)
        myuser.save()

        messages.success(request, "Your Account has been successfully created")
        return redirect('login') 
    return render(request, 'register.html')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.path.join(settings.BASE_DIR, 'garbage_classification_model2.pth')
try:
    classification_model = models.mobilenet_v2()
    num_ftrs = classification_model.classifier[1].in_features
    classification_model.classifier[1] = nn.Linear(num_ftrs, 2)  # Binary classification
    classification_model.load_state_dict(torch.load(model_path, map_location=device))
    classification_model.to(device)
    classification_model.eval()
except Exception as e:
    print(f"Error loading classification model: {e}")
    classification_model = None

try:
    yolo_model = YOLO("yolov8n.pt")  # Use custom path if applicable
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    yolo_model = None

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_map = {0: "recyclable", 1: "non-recyclable"}

class GarbageClassificationView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, *args, **kwargs):
        if classification_model is None or yolo_model is None:
            return Response({"error": "Model loading failed"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        file_obj = request.FILES.get('image')
        if not file_obj:
            return Response({"error": "No image provided"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            image = Image.open(file_obj).convert("RGB")
        except Exception as e:
            print(f"Error processing image: {e}")
            return Response({"error": "Invalid image format"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            results = yolo_model(image)
            detected_objects = []

            for result in results[0].boxes:
                try:
                    x1, y1, x2, y2 = map(int, result.xyxy[0])
                    cropped_image = image.crop((x1, y1, x2, y2))
                    transformed_image = transform(cropped_image).unsqueeze(0).to(device)
                    with torch.no_grad():
                        outputs = classification_model(transformed_image)
                        _, predicted = torch.max(outputs, 1)
                        label = class_map[predicted.item()]
                    detected_objects.append({
                        "bounding_box": [x1, y1, x2, y2],
                        "classification": label
                    })
                except Exception as e:
                    print(f"Error classifying object: {e}")
                    return Response({"error": "Classification error for detected object"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            return Response({"detections": detected_objects}, status=status.HTTP_200_OK)
        
        except Exception as e:
            print(f"Error in detection process: {e}")
            return Response({"error": "Detection or classification process failed"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

def submit_contact(request):
    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            form.save()
            return JsonResponse({"message": "Submission successful!"}, status=200)
        else:
            return JsonResponse({"errors": form.errors}, status=400)
    return JsonResponse({"error": "Invalid request method."}, status=405)
