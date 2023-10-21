import cv2
import glob
import random
import albumentations as A
from pandas.core.common import flatten
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):
    def __init__(self, image_paths, class_to_idx, transform=False):
        self.image_paths = image_paths
        self.transform = transform
        self.class_to_idx = class_to_idx
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = image_filepath.split("/")[-1].split("_")[0]
        label = self.class_to_idx[label]
        if self.transform:
            image = self.transform(image=image)["image"]
        
        return image, label


def dataset(train_data_path="", val_data_path="", test_data_path="", is_not_test=True):

    data_transforms = {
        "Train": 
            A.Compose([
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
        "Test": 
            A.Compose([
            A.Resize(224, 224),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
    }

    class_names = ['SeaRays','JellyFish','SeaUrchins','Otter','Penguin',
                    'Seahorse','Crabs','StarFish','Dolphin','Octopus']

    if is_not_test:
        val_image_paths = []
        train_image_paths = []

        for data_path in glob.glob(train_data_path + '/*'):
            train_image_paths.append(glob.glob(data_path + '/*.jpg'))
            
        for data_path in glob.glob(val_data_path + '/*'):
            val_image_paths.append(glob.glob(data_path + '/*.jpg'))

        train_image_paths = list(flatten(train_image_paths))
        val_image_paths = list(flatten(val_image_paths))
        random.shuffle(train_image_paths)
        idx_to_class = {i:j for i, j in enumerate(class_names)}
        class_to_idx = {value:key for key,value in idx_to_class.items()}
        image_datasets = {
            "Train" : ImageDataset(train_image_paths, class_to_idx, data_transforms["Train"]),
            "Validation" : ImageDataset(val_image_paths, class_to_idx, data_transforms["Test"]),
            }
        dataloaders = {
            "Train": DataLoader(image_datasets["Train"], batch_size=16, shuffle=True, num_workers=16),
            "Validation": DataLoader(image_datasets["Validation"], batch_size=16, shuffle=False, num_workers=16),
        }

        return dataloaders["Train"], dataloaders["Validation"]

    else:
        test_image_paths = []

        for data_path in glob.glob(test_data_path + '/*.jpg'):
            test_image_paths.append(data_path)

        test_image_paths = list(flatten(test_image_paths))
        idx_to_class = {i:j for i, j in enumerate(class_names)}
        class_to_idx = {value:key for key,value in idx_to_class.items()}
        image_datasets = {"Test" : ImageDataset(test_image_paths, class_to_idx, data_transforms["Test"])}
        dataloaders = {"Test": DataLoader(image_datasets["Test"], batch_size=16, shuffle=False, num_workers=16)}
    
        return dataloaders["Test"]