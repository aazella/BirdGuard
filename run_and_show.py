import cv2
from ultralytics import YOLO

# ===== 1. Load your trained model ==========================
MODEL_PATH = "pheasants.pt"   # <-- change if needed
model = YOLO(MODEL_PATH)

# ===== 2. Open webcam ======================================
cap = cv2.VideoCapture(0)   # 0 = default camera; try 1,2... if you have multiple

if not cap.isOpened():
    print("❌ Could not open webcam")
    raise SystemExit

print("✅ Webcam opened. Press 'q' to quit.")

# ===== 3. Live loop ========================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Run YOLO on the frame
    results = model(frame, conf=0.35, imgsz=640)

    # Get annotated frame with boxes + labels
    annotated_frame = results[0].plot()

    # Show it
    cv2.imshow("YOLOv8 Live", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===== 4. Cleanup ==========================================
cap.release()
cv2.destroyAllWindows()
