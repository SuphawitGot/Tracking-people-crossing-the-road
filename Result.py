from ultralytics import YOLO
import cv2
import supervision as sv
from ultralytics import solutions
import math 
import numpy as np
cap = cv2.VideoCapture('crosswalk.mp4')
mask = cv2.imread('mask.png')
model = YOLO("yolov8n.pt")
crosswalk_zone = np.array([[60, 700], [750, 550], [1100, 550], [300, 850]], np.int32) # Define the 4 corners of the crosswalk [Top-Left, Top-Right, Bottom-Right, Bottom-Left]


while True:
    success, img = cap.read()
    ImgRegion = cv2.bitwise_and(img,mask)
    result = model.track(ImgRegion, stream=True,classes=[0],verbose=False, tracker="bytetrack.yaml")
    people_in_crosswalk = 0
    for r in result:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) #tranfrom into integer 
            conf = math.ceil((box.conf[0]*100))/100
            
            if conf >= 0.60: # check accuracy before detection
                cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),1) 
                feetx = (x1+x2)//2
                feety = y2
                cv2.circle(img, (feetx, feety), 5, (0, 0, 255), cv2.FILLED) # dot on feet for easy
                is_inside = cv2.pointPolygonTest(crosswalk_zone, (feetx, feety), False)

                if is_inside >= 0:
                    people_in_crosswalk += 1
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3) # change to green color if in the zone 
    
    cv2.polylines(img, [crosswalk_zone], isClosed=True, color=(255,0,255), thickness=2) # draw the line of crosswalk
    
    
    if people_in_crosswalk > 0:
        cv2.putText(img,"Red Light",(50,50),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    else:
        cv2.putText(img,"Green Light",(50,50),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    
    cv2.imshow("image",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()




