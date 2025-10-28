
#nohup bash python train.py output.log 2>&1 &
from ultralytics import YOLO
import torch


torch.cuda.empty_cache()
model = YOLO("yolo11n.pt")

results = model.train(
    data="datasets/data.yaml",
    epochs=200,
    imgsz=640,
    batch=64,
    device="cuda",
    pretrained=True,
    cache=True,                         
    plots=True,                         
    save=True,
)



