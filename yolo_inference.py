from ultralytics import YOLO

model = YOLO('yolov8x')

# Using Weights from Model training with Roboflow 
# model = YOLO('models/yolo5_last.pt')

# Predicting on an image
# result = model.predict('input_videos/image.png', save=True)

# Predicting on a video
# result = model.predict('input_videos/input_video.mp4', conf=0.2, save=True)

# Tracking on a video
result = model.track('input_videos/input_video.mp4', conf=0.2, save=True)

print(result)
print("boxes:")
for box in result[0].boxes:
    print(box)

