from ultralytics import YOLO
import cv2

# ======= Load YOLOv8 model =======
model = YOLO('C:\\Users\\D Lawanya\\OneDrive\\Documents\\Sing language (N)\\best (4).pt')

# ======= Set confidence threshold =======
CONFIDENCE_THRESHOLD = 0.6  # 60%

# ======= Start webcam =======
cap = cv2.VideoCapture(0)  # Use 0 for default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting webcam... Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Run YOLO prediction
    results = model(frame)

    # Process results
    boxes = results[0].boxes
    for box in boxes:
        conf = float(box.conf[0])
        if conf >= CONFIDENCE_THRESHOLD:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw bounding box & label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show webcam frame
    cv2.imshow("Sign Language Detection", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ======= Release resources =======
cap.release()
cv2.destroyAllWindows()
