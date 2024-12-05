import torch
import cv2
import numpy as np
from PIL import Image

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Set confidence threshold
model.conf = 0.4  # Adjust as needed for accuracy

def extract_features(image, bbox):
    x_min, y_min, x_max, y_max = bbox
    object_img = image[int(y_min):int(y_max), int(x_min):int(x_max)]
    
    # Size Feature
    height, width = object_img.shape[:2]
    size_description = "small" if height * width < 5000 else "large"

    # Shape Feature (Approximate using contour detection)
    gray = cv2.cvtColor(object_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    shape_description = "undefined"
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * perimeter, True)
        if len(approx) == 3:
            shape_description = "triangle"
        elif len(approx) == 4:
            shape_description = "rectangle"
        elif len(approx) > 4:
            shape_description = "round"
    
    # Color Feature
    avg_color = object_img.mean(axis=0).mean(axis=0)
    color_description = "dark" if np.mean(avg_color) < 100 else "light"
    
    return size_description, shape_description, color_description

def generate_annotation(size, shape, color, label):
    return f"{size}, {color} {shape} object ({label})"

def annotate_image(image_path):
    # Load image
    img = Image.open(image_path)
    img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Run YOLOv5 model
    results = model(img)
    
    # Extract objects and annotate
    annotations = []
    for det in results.xyxy[0]:  # xyxy format of bounding boxes
        x_min, y_min, x_max, y_max, conf, cls = det
        bbox = [x_min.item(), y_min.item(), x_max.item(), y_max.item()]
        
        # Get label from YOLO model classes
        label = model.names[int(cls.item())]
        
        # Feature extraction
        size, shape, color = extract_features(img_cv2, bbox)
        
        # Generate annotation
        annotation = generate_annotation(size, shape, color, label)
        annotations.append((bbox, annotation))
        
        # Draw bounding box and annotation on image
        cv2.rectangle(img_cv2, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        cv2.putText(img_cv2, annotation, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display or save the annotated image
    annotated_image_path = "annotated_image.jpg"
    cv2.imwrite(annotated_image_path, img_cv2)
    print(f"Annotations: {annotations}")
    print(f"Annotated image saved as {annotated_image_path}")

# Path to your image
image_path = 'your_image.jpg'
annotate_image(image_path)
