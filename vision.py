import cv2
import numpy as np
import time

# Load model and label files
MODEL_PATH = "/home/jetson/jetson_detection/models/frozen_inference_graph.pb"
CONFIG_PATH = "/home/jetson/jetson_detection/models/ssd_mobilenet_v2_coco.pbtxt"
LABELS_PATH = "/home/jetson/jetson_detection/models/coco_names.txt"

# Load class labels
with open(LABELS_PATH, "r") as f:
    class_labels = f.read().strip().split("\n")

# Load pre-trained TensorFlow model
net = cv2.dnn.readNetFromTensorflow(MODEL_PATH, CONFIG_PATH)

# Set device to CUDA if available
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for default camera

# Set camera resolution (optional, adjust as needed)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Starting Object Detection...")

# Real-time loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    # Prepare image for the model
    blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)
    net.setInput(blob)

    # Run forward pass to get predictions
    detections = net.forward()

    # Loop through detected objects
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:  # Minimum confidence threshold
            class_id = int(detections[0, 0, i, 1])
            class_name = class_labels[class_id]

            # Get bounding box coordinates
            box_x, box_y, box_w, box_h = (detections[0, 0, i, 3:7] * 
                                           np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])).astype("int")

            # Draw bounding box and label
            label = f"{class_name}: {confidence * 100:.2f}%"
            cv2.rectangle(frame, (box_x, box_y), (box_w, box_h), (0, 255, 0), 2)
            cv2.putText(frame, label, (box_x, box_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow("Jetson Nano Object Detection", frame)

    # Break if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

