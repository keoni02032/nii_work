import torch as th
import pandas as pd
import numpy as np
import os
import os.path
import sys
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import torchvision.transforms as transforms
from PIL import Image
from albumentations import Compose, HorizontalFlip, RandomContrast, Crop, RandomBrightnessContrast, RandomCrop, Flip, RandomSizedCrop, OneOf, PadIfNeeded, Normalize, Resize, RandomCrop
from albumentations.pytorch.transforms import ToTensorV2
import torchvision
import cv2
from torchvision import datasets, models, transforms
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
import PIL

device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')

class MyDataset(Dataset):
    def __init__(self, file='', transforms=None):
        self.transform=transforms
        self.file = file
        data = self.data_parsing()
        self.df=pd.DataFrame(data, columns=self.classes())

    def classes(self):
        return list(range(0, 2))

    def __len__(self):
        return len(self.df)

    def data_parsing(self):
        file_name = self.file + '_lst.txt'
        print(file_name)
        data = open(file_name, 'r')
        lst = list()
        while True:
            tmp = data.readline()
            if not tmp:
                break
            tmp = tmp.split()
            lst.append([tmp[0], int(tmp[1])])
        return lst

    def __getitem__(self, idx):
        path = self.df.iloc[idx, 0]
        image = Image.open(path).convert('RGB')
        # image = cv2.imread(path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = PIL.Image.open(path)
        label = self.df.iloc[idx, 1]
        if self.transform:
            conv = transforms.Compose([
                transforms.Lambda(lambda x: x.convert('RGB') if len(x.split()) == 1 else x),
                transforms.RandAugment(num_ops=10)])
            image = conv(image)
            image = np.array(image)
            transforming = self.transform(image=image)
            image = transforming["image"]
        image = torchvision.transforms.functional.to_tensor(image)
        return image, label


def CreatingDataloaderTrainVal():
    import torchvision.transforms as T
    data_transforms_train = {'train': Compose([
                 Resize(286, 286, p=1.0),
                #  ToTensorV2(p=1.0),
                 RandomCrop(width=224, height=224),
                #  T.RandAugment(),
                 HorizontalFlip(p=0.5),                 
                 RandomBrightnessContrast(p=0.2),
                #  RandomContrast(p=0.2),
                 Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0)]),
    'val': Compose([Resize(224, 224, p=1.0),
                        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0), 
                        # ToTensorV2(p=1.0)
                        ])}
    
    data_dir = '256_ObjectCategories/data_train'
    device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')
    image_dataset_train = {x: MyDataset(x, data_transforms_train[x]) for x in ['train', 'val']}
    dataloader_train = {x: th.utils.data.DataLoader(image_dataset_train[x], batch_size=128, shuffle=True, num_workers=2) for x in ['train', 'val']}
    dataset_sizes_train = {x: len(image_dataset_train[x]) for x in ['train', 'val']}

    return dataloader_train, dataset_sizes_train

def CreatingDataloaderTest():
    data_transforms_test = {'test': Compose([Resize(224, 224, p=1.0),
                        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0), 
                        # ToTensorV2(p=1.0)
                        ])}

    data_dir = '256_ObjectCategories/data_train'
    device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')
    image_dataset_test = {x: MyDataset(x, data_transforms_test[x]) for x in ['test']}
    dataloader_test = {x: th.utils.data.DataLoader(image_dataset_test[x], batch_size=3, shuffle=True, num_workers=2) for x in ['test']}
    dataset_sizes_test = {x: len(image_dataset_test[x]) for x in ['test']}

    return dataloader_test, dataset_sizes_test