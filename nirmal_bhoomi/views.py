from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.authtoken.models import Token
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login as auth_login
from nirmal_bhoomi.serializers import UserSerializer, RegisterSerializer
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
from ultralytics import YOLO  # Import YOLO model for object detection

# Register API
'''class RegisterAPI(generics.GenericAPIView):
    serializer_class = RegisterSerializer

    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        
        # Ensure that the Token object is created properly
        token, _ = Token.objects.get_or_create(user=user)
        
        return Response({
            "user": UserSerializer(user, context=self.get_serializer_context()).data,
            "token": token.key
        })'''

# Load the garbage classification model
model_path = os.path.join(settings.BASE_DIR, 'garbage_classification_model2.pth')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the classification model architecture and weights
classification_model = models.mobilenet_v2()
num_ftrs = classification_model.classifier[1].in_features
classification_model.classifier[1] = nn.Linear(num_ftrs, 2)  # Binary classification
classification_model.load_state_dict(torch.load(model_path, map_location=device))
classification_model.to(device)
classification_model.eval()

# Load YOLO object detection model
yolo_model = YOLO("yolov8n.pt")  # or use a custom-trained YOLO model

# Define transformations (same as training)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the class mapping
class_map = {0: "recyclable", 1: "non-recyclable"}

class GarbageClassificationView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, *args, **kwargs):
        file_obj = request.FILES.get('image', None)
        if not file_obj:
            return Response({"error": "No image provided"}, status=status.HTTP_400_BAD_REQUEST)

        # Open the image
        image = Image.open(file_obj).convert("RGB")

        # Run object detection
        results = yolo_model(image)
        detected_objects = []

        for result in results[0].boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            
            # Crop the detected object from the image
            cropped_image = image.crop((x1, y1, x2, y2))
            
            # Transform the cropped image
            transformed_image = transform(cropped_image).unsqueeze(0)  # Add batch dimension

            # Run inference on the cropped image using the classification model
            with torch.no_grad():
                transformed_image = transformed_image.to(device)
                outputs = classification_model(transformed_image)
                _, predicted = torch.max(outputs, 1)
                label = class_map[predicted.item()]
                
            # Append result with bounding box and classification label
            detected_objects.append({
                "bounding_box": [x1, y1, x2, y2],
                "classification": label
            })

        # Return the results for each detected object
        return Response({"detections": detected_objects}, status=status.HTTP_200_OK)

# Render functions for frontend pages
def ai_intro(request):
    return render(request, 'ai-intro.html')

def ai(request):
    return render(request, 'ai.html')

def index(request):
    return render(request, 'index.html')

def login(request):
    if request.method == "POST":  # Corrected 'post' to 'POST'
        username = request.POST['username']
        pass1 = request.POST['pass1']

        # Authenticate the user
        user = authenticate(request, username=username, password=pass1)

        if user is not None:
            auth_login(request, user)
            #fname = user.first_name
            return render(request, "index_aft_log.html")
        else:
            messages.error(request, "Invalid Credentials")
            return redirect('login')  # Redirect to the sign-in page on failure

    return render(request, 'LOGIN.html')

def register(request):
    if request.method == "POST":  # Corrected 'post' to 'POST'
        username = request.POST['username']
        # fname = request.POST['fname']
        # lname = request.POST['lname']
        email = request.POST['email']
        pass1 = request.POST['pass1']
        pass2 = request.POST['pass2']

        # Check if passwords match before saving user
        if pass1 != pass2:
            messages.error(request, "Passwords do not match")
            return redirect('register')
        # Check if the username already exists
        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already taken. Please choose a different one.")
            return redirect('register')
        
        # Check if the email already exists (optional)
        if User.objects.filter(email=email).exists():
            messages.error(request, "An account with this email already exists.")
            return redirect('register')

        # Create user
        myuser = User.objects.create_user(username, email, pass1)
        # myuser.first_name = fname
        # myuser.last_name = lname
        myuser.save()

        messages.success(request, "Your Account has been successfully created")
        return redirect('login')  # Redirect to the sign-in page
    return render(request, 'register.html')