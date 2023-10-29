import torch
from ultralytics import YOLO
from argparse import ArgumentParser
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

parser = ArgumentParser()
parser.add_argument("--data", type=str, default="./Dataset/")
parser.add_argument("--model_path", type=str, default="../yolov8x-cls.pt")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--imgsz", type=int, default=224)
parser.add_argument("--batch", type=int, default=16)
args = parser.parse_args()

yolo = YOLO(model=args.model_path, task="classify")

device = 0 if torch.cuda.is_available() and args.device == "cuda" else "cpu"

freeze = []
for k, v in yolo.named_parameters():
    freeze.append(k)

yolo.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.bath, freeze=freeze[:-2], device=device)