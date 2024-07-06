"""
@File    : main.py
@Date    : 2024-06-13
@Author  : LiuTianSheng
@Software : yolo-learn
"""
from ultralytics import YOLO


model = YOLO("yolov8n.yaml").load("yolov8n.pt")

# Train the model
model = model.train(data="neudet.yaml", epochs=100, imgsz=200)

# model = YOLO("runs/detect/train/weights/best.pt")
# results = model('datasets/ImageSets/test/images/crazing_7.jpg')  # predict on an image


# model.tune(data="neudet.yaml", epochs=30, iterations=10, imgsz=200,optimizer="AdamW", plots=False, save=False, val=False)
