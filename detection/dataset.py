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

from PIL import Image
import os

class Data():
    def __init__(self, root="/content/detection/detection_db/", image_set="train", transform=None):
        self.transform = transform
        self.image_set = image_set
        self.root = root
        self.df = self.data_parsing()
        print(len(self.df))

    def __len__(self):
        return len(self.df)

    def data_parsing(self):
        data_train_label = os.listdir(f"{self.root}labels/{self.image_set}")
        data_train_images = os.listdir(f"{self.root}images/{self.image_set}")
        data_train = []
        for i in range(len(data_train_label)):
            if f'{data_train_label[i].split(".")[0]}.jpeg' in data_train_images:
                data = open(f"{self.root}labels/{self.image_set}/{data_train_label[i]}")
                while True:
                    tmp = data.readline()
                    if not tmp:
                        break
                    tmp = tmp.split(' ')
                    x_min = float(tmp[1]) - float(tmp[3]) / 2
                    y_min = float(tmp[2]) - float(tmp[4]) / 2
                    x_max = float(tmp[1]) + float(tmp[3]) / 2
                    y_max = float(tmp[2]) + float(tmp[4]) / 2
                    if x_min >= 0.0 and y_min >= 0.0 and x_max <= 1.0 and y_max <= 1.0:
                        if float(tmp[1]) >= 0.0 and float(tmp[2]) >= 0.0 and float(tmp[3]) <= 1. and float(tmp[4]) < 1.:
                            data_train.append([f"{self.root}images/{self.image_set}/{data_train_label[i].split('.')[0]}.jpeg", f"{self.root}labels/{self.image_set}/{data_train_label[i]}"])
        return pd.DataFrame(data_train, columns=range(0, 2))

    def __getitem__(self, index):
        path = self.df[0][index]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        path = self.df[1][index]
        bboxes = []
        category_ids = list()
        data = open(path)
        while True:
            tmp = data.readline()
            if not tmp:
                break
            tmp = tmp.split(' ')
            x_min = int(float(tmp[1]) * 1900 - float(tmp[3]) * 1900 / 2)
            y_min = int(float(tmp[2]) * 1060 - float(tmp[4]) * 1060 / 2)
            x_max = int(float(tmp[1]) * 1900 + float(tmp[3]) * 1900 / 2)
            y_max = int(float(tmp[2]) * 1060 + float(tmp[4]) * 1060 / 2)
            if x_min >= 0.0 and y_min >= 0.0 and x_max <= 1920 and y_max <= 1080:
                bboxes.append([x_min, y_min, x_max, y_max])
                category_ids.append(float(tmp[0]))
        bboxes = th.as_tensor(bboxes, dtype=th.float32)
        d = {}
        d['boxes'] = bboxes
        d['labels'] = category_ids
        if self.transform is not None:
            transformed = self.transform(image=image, bboxes=d['boxes'], class_labels=d['labels'])
            image = transformed["image"]
            d['boxes'] =  th.Tensor(transformed["bboxes"])
            d['labels'] = transformed["class_labels"]
        image = torchvision.transforms.functional.to_tensor(image)
        return image, d

def collate_fn(batch):
    return tuple(zip(*batch))

def CreatingDataloaderTrainVal():
    import torchvision.transforms as T
    data_transforms_train = {'train': Compose([
                 Resize(224, 224, p=1.0),
                #  ToTensorV2(p=1.0),
                 RandomCrop(width=176, height=176),
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