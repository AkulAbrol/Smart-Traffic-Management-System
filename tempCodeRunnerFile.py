from collections import defaultdict
from ultralytics import YOLO
import cv2

# Load the pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # You can replace 'yolov8n.pt' with a more powerful model like 'yolov8l.pt'

# Load the video
video_path = r"C:\Users\DELL\OneDrive\Desktop\cars_v\videoplayback.mp4" # Replace with the path to your video
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Define vehicle class IDs (for COCO dataset these are 'car', 'truck', 'bus', 'motorcycle')
vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']

# Get the video frame dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set the screen resolution for display purposes
screen_res = 1280, 720  # Adjust to your preferred screen resolution
scale_width = screen_res[0] / frame_width
scale_height = screen_res[1] / frame_height
scale = min(scale_width, scale_height)

window_width = int(frame_width * scale)
window_height = int(frame_height * scale)

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()

    # Break the loop if no frames are left
    if not ret:
        print("End of video.")
        break

    # Perform detection
    results = model(frame)

    # Extract the bounding boxes and class IDs
    detections = results[0].boxes
    classes = results[0].names

    # Variables to track the minimum and maximum coordinates for the large bounding box
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), 0, 0
    vehicle_count = 0
    class_count = defaultdict(int)  # To store the count of each detected vehicle class

    # Filter vehicle detections and calculate the big bounding box coordinates
    for i, cls in enumerate(detections.cls):
        class_name = classes[int(cls)]
        if class_name in vehicle_classes:
            vehicle_count += 1
            class_count[class_name] += 1

            # Get the bounding box coordinates for each vehicle
            box = detections.xyxy[i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box

            # Update the coordinates for the large bounding box
            min_x = min(min_x, x1)
            min_y = min(min_y, y1)
            max_x = max(max_x, x2)
            max_y = max(max_y, y2)

    # Draw the large bounding box around all detected vehicles
    if vehicle_count > 0:
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 3)  # Draw in blue with a thicker line

    # Prepare the label text with counts for each vehicle class
    label_text = ', '.join([f'{cls}: {count}' for cls, count in class_count.items()])

    # Put the label text inside the large bounding box
    cv2.putText(frame, f"Vehicles: {label_text}", (min_x, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Resize the frame for display
    resized_frame = cv2.resize(frame, (window_width, window_height))

    # Display the frame with the large bounding box and class counts
    cv2.imshow('Traffic Density', resized_frame)

    # Press 'q' to exit the video early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
