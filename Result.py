from ultralytics import YOLO
import cv2
import time

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)

# 1. Start a "stopwatch" before the loop begins
last_print_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, classes=[0], verbose=False)
    annotated = results[0].plot()
    
    cv2.imshow('Person Detection', annotated)

    for r in results:
        people_count = len(r.boxes) 
        
        # 2. Check the current time on the clock
        current_time = time.time()
        
        # 3. If the difference is 2 seconds or more, print the text!
        if current_time - last_print_time >= 2.0:
            print(f"People count: {people_count}")
            
            # 4. Reset the stopwatch
            last_print_time = current_time 
        
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()