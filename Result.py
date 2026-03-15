from ultralytics import YOLO
import cv2
import supervision as sv
from ultralytics import solutions
import math 

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("yolov8n.pt")

while True:
    success, img = cap.read()
    result = model(img, stream=True,classes=[0],verbose=False)
    for r in result:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) #tranfrom into integer
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            conf = math.ceil((box.conf[0]*100))/100
            cv2.putText(img,f'{conf}',(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,2,(255, 0, 0),3)
            if conf >= 0.70:
                print(conf)
                
    cv2.imshow("image",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break






