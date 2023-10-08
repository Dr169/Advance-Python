import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import gc
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


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
    
    
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3), nn.BatchNorm2d(64), nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self.make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self.make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self.make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self.make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)
        
    def make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
            
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
      
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


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
EPOCHS = 10
BATCH = 32
LR = 0.001
IMG_SIZE = 224
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
valid_loader = DataLoader(valid_dataset, batch_size=BATCH, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False)

resnet_model = ResNet(ResidualBlock, [3, 4, 6, 3]).to(device)
resnet_criterion = nn.CrossEntropyLoss()
resnet_optimizer = torch.optim.SGD(resnet_model.parameters(), lr=LR, weight_decay = 0.001, momentum = 0.9)

for epoch in range(EPOCHS):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = resnet_model(images)
        loss = resnet_criterion(outputs, labels)
        
        resnet_optimizer.zero_grad()
        loss.backward()
        resnet_optimizer.step()
        del images, labels, outputs
        torch.cuda.empty_cache()
        gc.collect()

    print (f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}')
            
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = resnet_model(images)
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
        outputs = resnet_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        del images, labels, outputs

    print(f'Accuracy of the network on the {len(test_dataset)} test images: {100 * correct / total:.2f} %')