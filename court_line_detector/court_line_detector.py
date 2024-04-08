import torch
import torchvision.transforms as transforms
import cv2
from torchvision import models
import numpy as np

class CourtLineDetector:
    def __init__(self, model_path):
        self.model = models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2) # 14 Key Points in X and Y
        self.model.load_state_dict(torch.load(model_path, map_location='cpu')) # Loads weights and Map to CPU 
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        # Just run on first frame for keypoints
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image_rgb).unsqueeze(0) # Apply transformations, and unsqueeze to add batch dimension (list of 1 image)
        with torch.no_grad():
            outputs = self.model(image_tensor) 
        keypoints = outputs.squeeze().cpu().numpy() # Squeeze removes batch dimension (list), and convert to numpy
        original_h, original_w = image.shape[:2]
        keypoints[::2] *= original_w / 224.0 # Every other element is X, so scale X
        keypoints[1::2] *= original_h / 224.0 # Every other element is Y, so scale Y

        return keypoints
    
    def draw_keypoints(self, image, keypoints):
        # Plot keypoints on the image
        for i in range(0, len(keypoints), 2): # Loop over keypoints (X, Y pairs)
            x = int(keypoints[i])
            y = int(keypoints[i+1])
            cv2.putText(image, str(i//2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2) # i//2 is the keypoint number
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1) # -1 is filled circle
        return image
    
    def draw_keypoints_on_video(self, video_frames, keypoints):
        output_video_frames = []
        for frame in video_frames:
            frame = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame)
        return output_video_frames