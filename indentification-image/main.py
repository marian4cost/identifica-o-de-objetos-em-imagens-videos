import cv2 as cv
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

imagePath = "cat01.jpg"
frame = cv.imread(imagePath)

results = model.track(frame, persist=True)
frameResult = results[0].plot()

cv.imshow("Image", frameResult)
cv.waitKey(0)
cv.destroyAllWindows()