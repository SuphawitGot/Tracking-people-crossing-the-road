from ultralytics import YOLO
import cv2


# load yolov8 model
model = YOLO('runs\\detect\\train12\\weights\\best.pt')

# load video
cap = cv2.VideoCapture(0)

ret = True
# read frames
while ret:
    ret, frame = cap.read()

    if ret:

        # detect objects
        # track objects
        results = model.track(frame, persist=True)

        # plot results
        # cv2.rectangle
        # cv2.putText
        frame_ = results[0].plot()

        # visualize
        cv2.imshow('frame', frame_)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
            
cap.release()
cv2.destroyAllWindows()




''' from ultralytics import YOLO

# 1. Load YOUR custom trained model instead of the base model
model = YOLO(r"runs\\detect\\train9\\weights\\best.pt") 

# 2. Perform object detection on a new image
results = model(r"D:\\Github repo\\PROJECT\AI\\computer vision\\japan-tokyo-shibuya-japanese-preview.jpg") 

# 3. Display the results
results[0].show() '''