from ultralytics import YOLO
import cv2
import threading
import time
import lgpio

# ================== GPIO SETUP ==================
LED_PIN = 17
chip = lgpio.gpiochip_open(0)
lgpio.gpio_claim_output(chip, LED_PIN)
lgpio.gpio_write(chip, LED_PIN, 0)

# ================== YOLO LOAD ===================
model = YOLO("pimodelv1.pt")

# ================== SHARED STATE ================
latest_frame = None
detected_now = 0            # 1 if current frame has ≥ conf LEGAL
lock = threading.Lock()

CONF_TRIGGER = 0.10
REQUIRED_ON_TIME = 1        # legal must be detected 1 second continuously

# Timing state
detection_start_time = None


# ================== YOLO THREAD =================
def yolo_thread():
    global latest_frame, detected_now

    while True:
        if latest_frame is None:
            time.sleep(0.01)
            continue

        lock.acquire()
        frame = latest_frame.copy()
        lock.release()

        small = cv2.resize(frame, (480, 480))

        results = model.predict(
            small,
            imgsz=480,
            conf=CONF_TRIGGER,    # keep consistent
            iou=0.45,
            device="cpu",
            verbose=False
        )

        detected_now = 0  # reset each cycle

        # ===== LEGAL = class 0 =====
        for det in results[0].boxes:
            cls = int(det.cls[0])
            conf = float(det.conf[0])
            if cls == 0 and conf >= CONF_TRIGGER:
                detected_now = 1
                break


# Start YOLO thread
threading.Thread(target=yolo_thread, daemon=True).start()


# ================== MAIN THREAD =================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not found!")
    exit()

print("🚀 LED ON only if LEGAL bird is detected for 1 full second")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    lock.acquire()
    latest_frame = frame.copy()
    lock.release()

    now = time.time()

    # ========== DETECTION TIMING LOGIC ==========
    if detected_now:
        if detection_start_time is None:
            detection_start_time = now

        if now - detection_start_time >= REQUIRED_ON_TIME:
            lgpio.gpio_write(chip, LED_PIN, 1)

    else:
        detection_start_time = None
        lgpio.gpio_write(chip, LED_PIN, 0)

    # Show camera
    cv2.imshow("BirdGuard Live", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# ================== CLEANUP ======================
lgpio.gpio_write(chip, LED_PIN, 0)
lgpio.gpiochip_close(chip)
cap.release()
cv2.destroyAllWindows()
print("👋 Done")
