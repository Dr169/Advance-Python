import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from torchvision import models
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

import os
import cv2
import time
import glob
import random
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt

from tqdm import tqdm
from pandas.core.common import flatten
from tempfile import TemporaryDirectory
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import precision_score, f1_score, recall_score, confusion_matrix

cudnn.benchmark = True
plt.ion()
writer = SummaryWriter('runs')

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("Running on CUDA!\n")
else:
    device = torch.device('cpu')
    print("Running on CPU!\n")

train_data_path = "./Dataset/Train/" 
test_data_path = "./Dataset/Validation/"

class_names = []
test_image_paths = []
train_image_paths = []

for data_path in glob.glob(train_data_path + '/*'):
    class_names.append(data_path.split('/')[-1]) 
    train_image_paths.append(glob.glob(data_path + '/*.jpg'))
    
for data_path in glob.glob(test_data_path + '/*'):
    test_image_paths.append(glob.glob(data_path + '/*.jpg'))

train_image_paths = list(flatten(train_image_paths))
test_image_paths = list(flatten(test_image_paths))
random.shuffle(train_image_paths)

idx_to_class = {i:j for i, j in enumerate(class_names)}
class_to_idx = {value:key for key,value in idx_to_class.items()}

class Dataset(Dataset):
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
    "Train" : Dataset(train_image_paths, data_transforms["Train"]),
    "Validation" : Dataset(test_image_paths, data_transforms["Validation"]),
}
dataloaders = {x:DataLoader(image_datasets[x], batch_size=16, shuffle=True) for x in ["Train", "Validation"]}
dataset_sizes = {x: len(image_datasets[x]) for x in ["Train", "Validation"]}
inputs, classes = next(iter(dataloaders["Validation"]))

img_grid = torchvision.utils.make_grid(inputs)
writer.add_image('Dataset sample', img_grid)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch [{epoch+1}/{num_epochs}]\n')

            train_acc = 0.0
            train_loss = 0.0
            class_labels = []
            class_preds = []

            for phase in ["Train", "Validation"]:
                if phase == "Train":
                    model.train()
                else:
                    model.eval()
                    
                running_loss = 0.0
                running_corrects = 0

                for i, (inputs, labels) in enumerate(tqdm(dataloaders[phase])):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    with torch.set_grad_enabled(phase == "Train"):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)

                        if phase == "Train":
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            class_probs_batch = [F.softmax(output, dim=0) for output in outputs]
                            class_preds.append(class_probs_batch)
                            class_labels.append(labels)

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)


                if phase == "Train":
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase] * 100

                if phase == "Train":
                    train_loss += epoch_loss
                    train_acc += epoch_acc

                print(f'{phase}: [Loss = {epoch_loss:.4f}] [Acc = {epoch_acc:.2f}%]\n')

                if phase == "Validation" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            class_preds = torch.cat([torch.stack(batch) for batch in class_preds])
            class_labels = torch.cat(class_labels)

            np_labels = class_labels.cpu().detach().numpy()
            np_preds = class_preds.cpu().detach().numpy()
            np_preds = np.argmax(np_preds, axis=1)

            cm = confusion_matrix(np_labels, np_preds)
            precision = precision_score(np_labels, np_preds, average="macro")
            recall = recall_score(np_labels, np_preds, average="macro")
            f1 = f1_score(np_labels, np_preds, average="macro")

            writer.add_scalar('Train accuracy', train_acc, epoch)
            writer.add_scalar('Train loss', train_loss, epoch)
            writer.add_scalar('Train precision', precision, epoch)
            writer.add_scalar('Train recall', recall, epoch)
            writer.add_scalar('Train f1', f1, epoch)

            print(f"Accuracy : {train_acc:.2f}%")
            print(f"Loss : {train_loss:.4f}")
            print(f"Confusion matrix :\n{cm}\n")
            print(f"Precision score : {precision:.4f}")
            print(f"Recall score : {recall:.4f}")
            print(f"F1 score : {f1:.4f}")
            
            classes = range(10)
            for i in classes:
                labels_i = class_labels == i
                preds_i = class_preds[:, i]
                writer.add_pr_curve(str(i+1), labels_i, preds_i, global_step=0)
                writer.close()

            print('_' * 50,"\n")
            
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best validation Acc: {best_acc:.2f}%')

        model.load_state_dict(torch.load(best_model_params_path))
        
    return model

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def visualize_model(model, num_images=10):
    was_training = model.training
    model.eval()
    images_so_far = 0

    with torch.no_grad():
        for inputs, labels in dataloaders["Validation"]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for i in range(inputs.size()[0]):
                images_so_far += 1
                imshow(inputs.cpu().data[i], f'predicted: {class_names[preds[i]]}')

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
                
        model.train(mode=was_training)

def visualize_model_predictions(model,img_path):
    was_training = model.training
    model.eval()

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = data_transforms["Validation"](image=img)["image"]
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)
        imshow(img.cpu().data[0], f'Predicted: {class_names[preds[0]]}')

        model.train(mode=was_training)

model_conv = models.resnet50(weights='IMAGENET1K_V2')

for param in model_conv.parameters():
    param.requires_grad = False

num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, len(class_names))
model_conv = model_conv.to(device)
criterion = nn.CrossEntropyLoss()
optimizer_conv = optim.Adam(model_conv.fc.parameters(), lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=10, gamma=0.1)
writer.add_graph(model_conv, inputs.to(device))
model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=20)

visualize_model(model_conv,1)
time.sleep(2)
random_imgs = random.sample(os.listdir("./Dataset/Test/"), 10)
for img in random_imgs:
    visualize_model_predictions(model_conv, img_path=f'./Dataset/Test/{img}')
    time.sleep(2)
    plt.close("all")

correct = {x:0 for x in class_names}
wrong  = {x:0 for x in class_names}
total = {x:0 for x in class_names}

for img_path in glob.glob("./Dataset/Test/*"):
    was_training = model_conv.training
    model_conv.eval()

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    label = img_path.split('/')[-1].split("_")[0]
    label = class_to_idx[label]
    img = data_transforms["Validation"](image=img)["image"]
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        outputs = model_conv(img)
        _, preds = torch.max(outputs, 1)
        model_conv.train(mode=was_training)

    if preds[0] == label:
        correct[class_names[label]] += 1
    elif preds[0] != label:
        wrong[class_names[label]] +=1
    total[class_names[label]] += 1

print(f"Test acc : {sum(correct.values())/sum(total.values())*100:.2f}%\n")
print(f"Correct : {correct}")
print(f"Wrong : {wrong}")
print(f"Total : {total}")