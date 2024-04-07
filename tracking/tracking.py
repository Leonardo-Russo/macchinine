import os
from ultralytics import YOLO

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Load the Model
model = YOLO('yolov8s.pt')

# Perform Tracking
results = model.track(source=0, show=True, tracker="bytetrack.yaml")