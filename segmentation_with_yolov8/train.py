import torch
from ultralytics import YOLO
from argparse import ArgumentParser
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

parser = ArgumentParser()
parser.add_argument("--data", type=str, default="./data.yaml")
parser.add_argument("--model_path", type=str, default="./yolov8x-seg.pt")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--batch", type=int, default=4)
parser.add_argument("--freeze", type=int, default=0)
parser.add_argument("--export", type=bool, default=True)
parser.add_argument("--format", type=str, default="onnx")
args = parser.parse_args()

model = YOLO(model=args.model_path, task="segment")

device = 0 if torch.cuda.is_available() and args.device == "cuda" else "cpu"

want_to_freeze = [f'model.{x}.' for x in range(args.freeze)]
freezed_layers = []
for k, v in model.named_parameters():
    if any(x in k for x in want_to_freeze):
        v.requires_grad = False
        freezed_layers.append(k)

model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, freeze=freezed_layers, device=device)

if args.export:
    model.export(format=args.format, dynamic=True)
    print(f"Model exported to {args.format} format")