import torch as th
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
from dataset import CreatingDataloaderTrainVal, CreatingDataloaderTest

device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')

class Models():
    def __init__(self):
        pass

    def CreatingModel(self):
        model_ft = models.resnet18(pretrained=False)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 256)
        model_ft = model_ft.to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=1e-5)
        optimizer_ft = optim.AdamW(model_ft.parameters(), lr=0.0005, weight_decay=0.01)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.5)
        return model_ft, criterion, optimizer_ft, exp_lr_scheduler

    def LoadModel(self):
        model_ft = models.resnet18(pretrained=False, num_classes=256)
        model_ft.load_state_dict(th.load('/content/classification/model_0.pth'))
        model_ft = model_ft.to(device)
        return model_ft

    def CreatingTimmModel(self):
        dataloader_train, dataset_sizes_train = CreatingDataloaderTrainVal()
        model = timm.create_model('resnet18', pretrained=False, num_classes=256)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=1e-5)
        learning_rate = th.tensor(0.02)
        optimizer_ft = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0005)
        exp_lr_scheduler = lr_scheduler.OneCycleLR(optimizer_ft, max_lr=learning_rate, steps_per_epoch=len(dataloader_train['train']), epochs=80)
        return model, criterion, optimizer_ft, exp_lr_scheduler

    def LoadTimmModel(self):
        model = timm.create_model('resnet18', pretrained=False, num_classes=256)
        model.load_state_dict(th.load('/content/classification/model_0.pth'))
        # model.load_state_dict(torch.load(args.model))
        model = model.to(device)
        return model