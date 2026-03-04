from ultralytics import YOLO


if __name__ == '__main__':
# Load a pretrained YOLO26n model
    model = YOLO("yolov8n.pt")

# Train the model on the COCO8 dataset for 100 epochs
    train_results = model.train(
        data=r"D:\\Github repo\\PROJECT\AI\\computer vision\\dataset\\data.yaml",  # Path to dataset configuration file
        epochs=70,  # Number of training epochs
        imgsz=640,  # Image size for training
        device="0",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
    )

# Evaluate the model's performance on the validation set
    metrics = model.val()

# Pass the folder path (no file name at the end)
    results = model(r"D:\\Github repo\\PROJECT\\AI\\computer vision\\dataset\\test\\images\\GX010023_frame_03012_right_jpg.rf.23482362a007030fa3c43d038890ce0c.jpg") 

    results[0].show()  # Display results

# Export the model to ONNX format for deployment
    path = model.export(format="onnx")  # Returns the path to the exported model