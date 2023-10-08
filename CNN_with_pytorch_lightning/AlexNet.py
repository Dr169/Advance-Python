import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import cv2
import glob
import random
import albumentations as A
from pandas.core.common import flatten
from albumentations.pytorch import ToTensorV2


if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Running on CUDA!")
else:
    device = torch.device('cpu')
    print("Running on CPU!")


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))
      
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


classes = []
test_image_paths = []
train_image_paths = []

train_data_path = "./Dataset/Train/" 
test_data_path = "./Dataset/Test/"

for data_path in glob.glob(train_data_path + '/*'):
    classes.append(data_path.split('/')[-1]) 
    train_image_paths.append(glob.glob(data_path + '/*'))
    
for data_path in glob.glob(test_data_path + '/*'):
    test_image_paths.append(glob.glob(data_path + '/*'))

train_image_paths = list(flatten(train_image_paths))
random.shuffle(train_image_paths)
train_image_paths = train_image_paths[:int(0.8*len(train_image_paths))]
valid_image_paths = train_image_paths[int(0.8*len(train_image_paths)):]

test_image_paths = list(flatten(test_image_paths))

idx_to_class = {i:j for i, j in enumerate(classes)}
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
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        
        return image, label
    

CLASSES = 10
EPOCHS = 40
BATCH = 128
LR = 0.01
IMG_SIZE = 227
TRAIN_TRANSFORMS =  A.Compose([
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=360, p=0.5),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.MultiplicativeNoise(multiplier=[0.5,2], per_channel=True, p=0.2),
    A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ToTensorV2(),
])
TEST_TRANSFORMS = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ToTensorV2(),
])

train_dataset = Dataset(train_image_paths, TRAIN_TRANSFORMS)
valid_dataset = Dataset(valid_image_paths, TEST_TRANSFORMS)
test_dataset = Dataset(test_image_paths, TEST_TRANSFORMS)

print(f"Train size: {len(train_dataset)}\nValid size: {len(valid_dataset)}\nTest size: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False)

alexnet_model = AlexNet(CLASSES).to(device)
alexnet_criterion = nn.CrossEntropyLoss()
alexnet_optimizer = torch.optim.SGD(alexnet_model.parameters(), lr=LR, weight_decay = 0.005, momentum = 0.9)

for epoch in range(EPOCHS):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = alexnet_model(images)
        loss = alexnet_criterion(outputs, labels)

        alexnet_optimizer.zero_grad()
        loss.backward()
        alexnet_optimizer.step()

    print (f'Epoch [{epoch+1}/{EPOCHS}], Loss: { loss.item():.4f}')

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = alexnet_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs
    
        print(f'Accuracy of the network on the {len(valid_dataset)} validation images: {100 * correct / total:.2f} %')  

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = alexnet_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        del images, labels, outputs

    print(f'Accuracy of the network on the {len(test_dataset)} test images: {100 * correct / total:.2f} %')   