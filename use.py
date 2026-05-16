from ultralytics import YOLO
model = YOLO("yolo11n.pt")  # 加载训练好权重
model("datasets\\VOC2007_yolo_small\\images\\train\\000019.jpg", show=True) # 直接识别