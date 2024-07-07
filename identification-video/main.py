import cv2 as cv
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

videoPath = "01.mp4"
cap = cv.VideoCapture(videoPath)
ret, frame = cap.read()

frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

while ret:
    ret, frame = cap.read()
    frame = cv.resize(frame, (frame_width, frame_height))

    results = model.track(frame, persist=True)
    frameResult = results[0].plot()

    cv.imshow("frame", frameResult)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()