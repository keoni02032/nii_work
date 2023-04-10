import torch as th
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, lraspp_mobilenet_v3_large, deeplabv3_resnet101,deeplabv3_resnet50
import torchvision
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')

class Models():
    def __init__(self):
        pass

    def CreatingModel(self):
        model = deeplabv3_mobilenet_v3_large(False, num_classes=5, progress=True)
        optimizer_ft = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.0005)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=33, gamma=0.1)
        crit = th.nn.CrossEntropyLoss(label_smoothing=1e-5)
        model.to(device)
        return model, crit, optimizer_ft, exp_lr_scheduler

    def LoadModel(self):
        model_ft = deeplabv3_mobilenet_v3_large(False, num_classes=5)
        model_ft.load_state_dict(th.load('/content/segmentation/model_1.pth'))
        model_ft = model_ft.to(device)
        return model_ft

    def CreateFTModel(self, keep_feature_extract=True):
        model_ft = deeplabv3_resnet101(pretrained=True, progress=True)
        model_ft.classifier = DeepLabHead(2048, 5)
        optimizer_ft = optim.AdamW(model_ft.parameters(), lr=0.0001, weight_decay=0.0005)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=35, gamma=0.1)
        crit = th.nn.CrossEntropyLoss(label_smoothing = 1e-5)
        model_ft.to(device)
        return model_ft, crit, optimizer_ft, exp_lr_scheduler   