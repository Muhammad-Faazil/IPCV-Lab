from ultralytics import YOLO
#import cv2
model=YOLO('yolov8x.pt')
results=model(r"D:\4SF23CI052-DHWANI\IPCV\Untitled.jpg")
results[0].show()
