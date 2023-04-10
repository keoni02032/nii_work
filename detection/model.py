import torch as th
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, ssd300_vgg16
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
# from focal_loss_with_smoothing import FocalLossWithSmoothing

device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with th.no_grad():
            # true_dist = pred.data.clone()
            true_dist = th.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            # true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return (-true_dist * pred).max(dim=1).values
    

class FocalLossWithLabelSmoothing(nn.Module):
    def __init__(self, gamma=2., reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.CE = LabelSmoothingLoss(classes=2, smoothing=0.1)

    def forward(self, inputs, targets):
        CE_loss = self.CE(inputs, targets)
        pt = th.exp(-CE_loss)
        F_loss = ((1 - pt)**self.gamma) * CE_loss
        if self.reduction == 'sum':
            return F_loss.sum()
        elif self.reduction == 'mean':
            return F_loss.mean()

class Models():
    def __init__(self):
        pass

    def CreatingModel(self):
        model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(False, num_classes=2)
        criteria = FocalLossWithLabelSmoothing(2)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        model.to(device)
        return model, optimizer, criteria, exp_lr_scheduler

    def LoadModel(self):
        model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(False, num_classes=2)
        model.load_state_dict(th.load('/content/detection/model_1.pth'))
        model = model.to(device)
        return model

    def CreateFTModel(self):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        num_classes = 2
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        criteria = FocalLossWithLabelSmoothing(num_classes)
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0005)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
        model.to(device)
        return model, optimizer, criteria, exp_lr_scheduler

    def CreateFTModelAndDiffBackbone(self):
        backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
        backbone.out_channels = 1280
        criteria = FocalLossWithLabelSmoothing(2)
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
        model = FasterRCNN(backbone, num_classes=2, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        model.to(device)
        return model, optimizer, criteria, exp_lr_scheduler