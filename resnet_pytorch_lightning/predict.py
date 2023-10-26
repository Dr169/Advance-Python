import os
import cv2
import glob
import torch
import albumentations as A
import torch.backends.cudnn as cudnn

from argparse import ArgumentParser
from lightning_model import LightningModel
from albumentations.pytorch import ToTensorV2

if __name__ == "__main__":
    cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    parser = ArgumentParser()
    parser.add_argument("--predict_data", type=str, default="./Dataset/Predict")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/best.ckpt")
    args = parser.parse_args()

    model = LightningModel.load_from_checkpoint(checkpoint_path=args.checkpoint_path)

    transform =  A.Compose([A.Resize(224, 224), A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ToTensorV2()])
    
    predicted = []

    if os.path.isdir(args.predict_data):
        for img in glob.glob(args.predict_data + "/*.jpg"):
            image = cv2.imread(img)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = transform(image=image)["image"].to(device)
            y_hat = model(image.unsqueeze(0)).argmax()
            predicted.append((img, model.class_names[y_hat]))
    else:
        image = cv2.imread(args.predict_data)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transform(image=image)["image"].to(device)
        y_hat = model(image.unsqueeze(0)).argmax()
        predicted.append((args.predict_data, model.class_names[y_hat]))


    print(*predicted , sep="\n")