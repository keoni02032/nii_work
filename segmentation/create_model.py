import torch as th
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')

class Models():
    def CreatingModel(self):
        model = smp.DeepLabV3Plus(encoder_weights='imagenet', classes=5, activation='softmax')
        model.segmentation_head[2].activation = nn.Identity()
        optimizer_ft = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.0005)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=33, gamma=0.1)
        crit = th.nn.CrossEntropyLoss(label_smoothing=1e-5)
        model.to(device)
        return model, crit, optimizer_ft, exp_lr_scheduler

    def LoadModel(self):
        model = smp.DeepLabV3Plus(encoder_weights='imagenet', classes=5, activation='softmax')
        model.segmentation_head[2].activation = nn.Identity()
        model.load_state_dict(th.load('/content/segmentation/model_1.pth'))
        return model_1