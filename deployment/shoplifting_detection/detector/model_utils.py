import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms
from torchvision.models.video import r3d_18
import os
from django.conf import settings

# Constants
NUM_FRAMES = 16
IMG_SIZE = 112
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load model once as global variable to avoid reloading on each request
def load_model():
    """Load the trained R3D-18 model"""
    model = r3d_18(pretrained=False)
    model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(model.fc.in_features, 2))

    # Load the saved weights
    model_path = os.path.join(settings.BASE_DIR, 'models', 'best_3D_CNN_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    return model


# Initialize model globally
MODEL = load_model()


def sample_uniform_frames(video_path, num_frames=NUM_FRAMES):
    """Uniformly sample frames from a video"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames >= num_frames:
        frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    else:
        frame_indices = list(range(total_frames)) + [total_frames - 1] * (num_frames - total_frames)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            frame = frames[-1] if frames else np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return np.array(frames)


def predict_video(video_path):
    """Predict shoplifting for a given video"""
    # Define transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    # Sample frames from video
    frames = sample_uniform_frames(video_path)

    # Transform frames
    frames = [transform(frame) for frame in frames]
    frames = torch.stack(frames).permute(1, 0, 2, 3)  # Shape: (C, T, H, W)
    frames = frames.unsqueeze(0)  # Add batch dimension: (1, C, T, H, W)
    frames = frames.to(device)

    # Make prediction
    with torch.no_grad():
        outputs = MODEL(frames)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()

    # Return results
    class_labels = {0: "Normal", 1: "Shoplifting"}
    confidence = probabilities[0][predicted_class].item() * 100

    return {
        'prediction': class_labels[predicted_class],
        'confidence': round(confidence, 2),
        'probabilities': {
            'normal': round(probabilities[0][0].item() * 100, 2),
            'shoplifting': round(probabilities[0][1].item() * 100, 2)
        }
    }
