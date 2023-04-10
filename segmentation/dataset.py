import torch as th
import pandas as pd
import numpy as np
import os
import os.path
import sys
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import torchvision.transforms as transforms
from albumentations import Compose, RandomBrightnessContrast, RandomCrop, Flip, RandomSizedCrop, OneOf, PadIfNeeded, Normalize, Resize
from PIL import Image
import cv2
import torchvision

device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')

from PIL import Image

COLORS = [
    [150,  95,  60],
    [195, 195, 195],
    [ 88,  88,  88],
    [  0, 120,   0],
    [255, 255,   0],
]

class PascalVOCSearchDataset():
    def __init__(self, root="~/data/pascal_voc", image_set="train", transform=None):
        self.transform = transform
        self.root = root
        self.image_set = image_set
        data = self.data_parsing()
        self.df = pd.DataFrame(data, columns=self.classes())

    def classes(self):
        return list(range(0, 2))

    def __len__(self):
        return len(self.df)

    def data_parsing(self):
        file_name = self.image_set + '_lst.txt'
        print(file_name)
        data = open(f'{self.root}/lists/{file_name}')
        lst = list()
        while True:
            tmp = data.readline()
            if not tmp:
                break
            tmp = str(tmp.split())
            lst.append([f'{self.root}/images/Img_{tmp[2:5]}.jpeg', f'{self.root}/masks/Mask_{tmp[2:5]}.png'])
        return lst

    @staticmethod
    def _convert_to_segmentation_mask(mask):
        height, width = mask.shape[:2]
        segmentation_mask = np.zeros((height, width, len(COLORS)), dtype=np.float32)
        for label_index, label in enumerate(COLORS):
            segmentation_mask[:, :, label_index] = np.all(mask == label, axis=-1).astype(float)
        return segmentation_mask

    def __getitem__(self, index):
        path = self.df.iloc[index, 0]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        path = self.df.iloc[index, 1]
        mask = cv2.imread(path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = self._convert_to_segmentation_mask(mask)
        if self.transform is not None:
            
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        image,mask = torchvision.transforms.functional.to_tensor(image), th.from_numpy(mask)
        mask = mask.permute(2,0,1)
        return image, mask

def CreatingDataloaderTrainVal():
    train_augs = Compose([Resize(1080, 1920),
                 PadIfNeeded(448,448),
                 RandomBrightnessContrast(p=0.2),
                 OneOf([
                        RandomCrop(256,256, p=0.2),
                        RandomSizedCrop((224,448),256,256)
                 ], p =1),
                 Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0)])

    val_augs = Compose([Resize(1088, 1920),Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0)])
    train_dataset = PascalVOCSearchDataset('/content/segmentation/synthetic', transform = train_augs)
    val_dataset = PascalVOCSearchDataset('/content/segmentation/synthetic', 'val', transform = val_augs)
    batch_size = 5
    dataloader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers = 2)
    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = 2) 
    dataloader = {'train': dataloader_train, 'val': dataloader_val}

    return dataloader