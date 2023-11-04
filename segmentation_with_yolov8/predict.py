import glob
import torch
from ultralytics import YOLO
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--data", type=str, default="../datasets/segmentation/test")
parser.add_argument("--model_path", type=str, default="./runs/segment/train/weights/best.pt")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--imgsz", type=int, default=640)
args = parser.parse_args()

model = YOLO(args.model_path, task="segment")

device = 0 if torch.cuda.is_available() and args.device == "cuda" else 'cpu'

data_set = glob.glob(f"{args.data}/*.jpg")
for img in data_set:
    model.predict(img, save=True, imgsz=args.imgsz, device=device)