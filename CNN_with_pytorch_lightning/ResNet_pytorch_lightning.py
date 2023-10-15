import torch
import torch.nn as nn
import lightning.pytorch as pl
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from torchvision import models
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.functional import precision, recall, f1_score

import cv2
import glob
import random
import operator
import collections
import albumentations as A

from functools import reduce
from pandas.core.common import flatten
from albumentations.pytorch import ToTensorV2


cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
device = "cuda:0" if torch.cuda.is_available() else "cpu"


class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=False):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = image_filepath.split('/')[-2]
        label = class_to_idx[label]
        if self.transform:
            image = self.transform(image=image)["image"]
        
        return image, label


class LightningModel(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet50(weights='IMAGENET1K_V2')
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (torch.argmax(y_hat, dim=1) == y).float().mean() * 100
        self.precision = precision(y_hat, y,task="multiclass", num_classes=10)
        self.recall = recall(y_hat, y,task="multiclass", num_classes=10)
        self.f1 = f1_score(y_hat, y,task="multiclass", num_classes=10)
        self.log_dict({
            'train_loss': loss,
            'train_acc': acc,
            'train_precision': self.precision,
            'train_recall': self.recall,
            'train_f1': self.f1
        }, on_step=False, on_epoch=True, logger=True)

        return {
            'loss': loss,
            'acc': acc,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1
        }
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (torch.argmax(y_hat, dim=1) == y).float().mean() * 100
        self.precision = precision(y_hat, y,task="multiclass", num_classes=10)
        self.recall = recall(y_hat, y,task="multiclass", num_classes=10)
        self.f1 = f1_score(y_hat, y,task="multiclass", num_classes=10)
        self.log_dict({
            'val_loss': loss,
            'val_acc': acc,
            'val_precision': self.precision,
            'val_recall': self.recall,
            'val_f1': self.f1
        }, on_step=False, on_epoch=True, logger=True)

        return {
            'loss': loss,
            'acc': acc,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1
        }
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }


def dataset(train_path, val_path):
    train_data_path =  train_path
    val_data_path = val_path

    class_names = []
    test_image_paths = []
    train_image_paths = []

    for data_path in glob.glob(train_data_path + '/*'):
        class_names.append(data_path.split('/')[-1]) 
        train_image_paths.append(glob.glob(data_path + '/*.jpg'))
        
    for data_path in glob.glob(val_data_path + '/*'):
        test_image_paths.append(glob.glob(data_path + '/*.jpg'))

    train_image_paths = list(flatten(train_image_paths))
    test_image_paths = list(flatten(test_image_paths))
    random.shuffle(train_image_paths)


    return class_names, train_image_paths , test_image_paths


def accuracy(test_path, model):
    acc_data = {x:{"correct":0,"wrong":0,"total":0, "acc":0.0} for x in class_names}

    for img_path in glob.glob(f"{test_path}/*"):
        was_training = model.training
        model.eval()

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = img_path.split('/')[-1].split("_")[0]
        label = class_to_idx[label]
        img = data_transforms["Validation"](image=img)["image"]
        img = img.unsqueeze(0)
        img = img.to(device)

        with torch.no_grad():
            outputs = model(img)
            _, preds = torch.max(outputs, 1)
            model.train(mode=was_training)

        if preds[0] == label:
            acc_data[class_names[label]]["correct"] += 1
        elif preds[0] != label:
            acc_data[class_names[label]]["wrong"] +=1
        acc_data[class_names[label]]["total"] += 1
        acc_data[class_names[label]]["acc"] = float(f"{acc_data[class_names[label]]['correct']/acc_data[class_names[label]]['total'] * 100:.2f}")

    result = dict(reduce(operator.add,map(collections.Counter, list(acc_data.values()))))

    print(f"Test Accuracy : {result['correct']/result['total']*100:.2f}%\n")
    print("Data Information :")
    for key in acc_data.keys():
        print(f"\t{key} :")
        for k,v in acc_data[key].items():
            print(f"\t\t{k} : {v}")


class_names, train_image_paths, test_image_paths = dataset(
    "./Dataset/Train/",
    "./Dataset/Validation/"
    )
idx_to_class = {i:j for i, j in enumerate(class_names)}
class_to_idx = {value:key for key,value in idx_to_class.items()}
data_transforms = {
    "Train": A.Compose([
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=360, p=0.5),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.MultiplicativeNoise(multiplier=[0.5,2], per_channel=True, p=0.2),
        A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.Resize(224, 224),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]),
    "Validation": A.Compose([
        A.Resize(224, 224),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]),
}
image_datasets = {
    "Train" : ImageDataset(train_image_paths, data_transforms["Train"]),
    "Validation" : ImageDataset(test_image_paths, data_transforms["Validation"]),
    }
dataloaders = {
    "Train": DataLoader(image_datasets["Train"], batch_size=16, shuffle=True, num_workers=16),
    "Validation": DataLoader(image_datasets["Validation"], batch_size=16, shuffle=False, num_workers=16)
    }
dataset_sizes = {x : len(image_datasets[x]) for x in ["Train","Validation"]}       
inputs, classes = next(iter(dataloaders["Validation"]))


logger = TensorBoardLogger("logs", name="CNN_model")
model_conv = LightningModel(num_classes=10)
trainer = pl.Trainer(max_epochs=20, logger=logger)
trainer.fit(model_conv.to(device), dataloaders["Train"], dataloaders["Validation"])

accuracy("./Dataset/Test/",model_conv.to(device))