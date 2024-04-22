import os
from ultralytics import YOLO
import pickle

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Load the Model
model = YOLO('yolov8s.pt')

# Perform Tracking
results = model.track(source=r'tracking\test.mp4', show=True, tracker="bytetrack.yaml")

for result in results:

    # coords_norm = result.boxes.xywhn[0].tolist()
    # conf = result.boxes.conf[0].tolist()
    # class_id = int(result.boxes.cls[0].tolist())
    print(result.boxes)
    with open('results_boxes.pkl', 'wb') as f:
        pickle.dump(results.boxes, f)
    

    break