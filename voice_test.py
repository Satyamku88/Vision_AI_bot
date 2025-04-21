from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")  # You can also try "yolov8s.pt" for better accuracy
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(frame, verbose=False)
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = r.names[cls]
            print("Detected:", label)

    cv2.imshow("YOLO Test", frame)
    if cv2.waitKey(1) == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
