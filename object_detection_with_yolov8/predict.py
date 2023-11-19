import cv2
import glob
import torch
import albumentations as A
from ultralytics import YOLO
from argparse import ArgumentParser
from albumentations.pytorch import ToTensorV2


parser = ArgumentParser()
parser.add_argument("--data", type=str, default="./dataset")
parser.add_argument("--model_path", type=str, default="best.pt")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--batch", type=int, default=4)
args = parser.parse_args()

device = 0 if torch.cuda.is_available() and args.device == "cuda" else 'cpu'

transform = A.Compose([ToTensorV2()])
data_set = glob.glob(f"{args.data}/*.jpg")
batched_data = []
batch = torch.zeros(0)
for index, img in enumerate(data_set,1):
    image = cv2.imread(img)
    image = cv2.resize(image, (args.imgsz, args.imgsz)) / 255.0
    image = transform(image=image)["image"]
    batch = torch.cat((batch, image.unsqueeze(0)), dim=0)

    if index % args.batch == 0:
        batched_data.append(batch)
        batch = torch.zeros(0)

if len(batch) != 0:
    batched_data.append(batch)


model = YOLO(args.model_path)

for idx, batch in enumerate(batched_data,1):
    model.predict(batch, save=True, device=device, project="./runs/detect/predict/", name=f"batch_{idx}/")