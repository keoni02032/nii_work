import torch as th
import torchvision
from torchvision.ops import focal_loss
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from torch.utils.tensorboard import SummaryWriter
from model import Models
from dataset import CreatingDataloaderTrainVal, collate_fn
import tqdm
from tqdm import tqdm
import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision

device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')

def train_model(model, optimizer_ft, crit, exp_lr_scheduler, num_epochs=40):
    next_desc = "first epoch"
    pred = []
    targ = []
    best_map = 0
    best_map50 = 0
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                all_losses = []
                all_losses_dict = []
                for img, target in tqdm(dataloader[phase], desc = next_desc):
                    # target = [{k: th.tensor(v).to(device) for k, v in t.items()} for t in target]
                    tmp = list()
                    for i in range(len(img)):
                        tmp.append(img[i].to(device))
                    predict = model(tmp, target)   
                    losses = sum(loss for loss in predict.values())
                    losses_dict_append = {k: v.item() for k, v in loss_dict.items()}
                    loss_value = losses.item()

                    all_losses.append(loss_value)
                    all_losses_dict.append(losses_dict_append)
                    losses.backward()
                    optimizer_ft.step()
                    optimizer_ft.zero_grad()
                    bbox_regression = float(predict['bbox_regression'])
                    classification = float(predict['classification'])
                    # loss_classifier = float(predict['loss_classifier'])
                    # loss_box_reg = float(predict['loss_box_reg'])
                    # loss_objectness = float(predict['loss_objectness'])
                    # loss_rpn_box_reg = float(predict['loss_rpn_box_reg'])
                exp_lr_scheduler.step()
                # writer.add_scalar('train loss', rloss, epoch)
                # writer.add_scalar('train MIoU', miou, epoch)
            else:
                model.eval()
                with th.no_grad():
                    focal_loss = 0
                    for img, target in tqdm(dataloader[phase], desc = next_desc):
                        target = [{k: th.tensor(v).to(device) for k, v in t.items()} for t in target]
                        tmp = list()
                        for i in range(len(img)):
                            tmp.append(img[i].to(device))
                        for i in range(len(target)):
                            target[i]['boxes'] = target[i]['boxes'].int()
                            target[i]['labels'] = target[i]['labels'].long()
                        predict = model(tmp)
                        loss = crit(predict[0]['labels'].float().unsqueeze(1), target[0]['labels'].float().unsqueeze(1))
                        for i in range(len(pred)):
                            p = pred[i]['labels'].float().unsqueeze(0)
                            t = target[1]['labels'].float().unsqueeze(0)
                            focal_loss += crit(p, t).item()
                        for i in range(len(img)):
                            target[i]['boxes'] = target[i]['boxes'].float().cpu()
                            target[i]['labels'] = target[i]['labels'].cpu()
                            predict[i]['boxes'] = predict[i]['boxes'].float().cpu()
                            predict[i]['labels'] = predict[i]['labels'].cpu()
                            predict[i]['scores'] = predict[i]['scores'].cpu()

                        for i in range(len(predict)):
                            pred.append(predict[i])
                            targ.append(target[i])
                    print('loss: ', focal_loss / (len(dataloader[phase]) * 3))
                    metric = MeanAveragePrecision()
                    metric.update(pred, targ)
                    map_50 = metric.compute()['map_50']
                    map = metric.compute()['map']
                    pred = []
                    targ = []
                    if best_map50 < map_50 and best_map < map:
                        th.save(model.state_dict(), f"./model_{epoch + 1}.pth")
                    # writer.add_scalar('valtrain loss', rloss, epoch)
                    # writer.add_scalar('val MIoU', miou, epoch)
            if phase == 'train':
                next_desc = f"Epoch of {phase}: [{epoch+1}], lr: [{optimizer_ft.param_groups[0]['lr']}], classification: [{np.mean(all_losses)}]"
                # next_desc = f"Epoch of {phase}: [{epoch+1}], loss_classifier: [{loss_classifier}], loss_box_reg: [{loss_box_reg}], loss_objectness: [{loss_objectness}], loss_rpn_box_reg: [{loss_rpn_box_reg}] "
            else:
                next_desc = f'Epoch of {phase}: [{epoch+1}]  map_50: [{map_50}], map: [{map}]'
            if epoch == num_epochs - 1 and phase == 'val':
                print(f'Epoch of {phase}: [{epoch+1}] map_50: [{map_50}], map: [{map}]')

if __name__ == "__main__":
    dataloader = CreatingDataloaderTrainVal()
    model = Models()
    model, optimizer_ft, crit, exp_lr_scheduler = model.CreatingModel()
    model = train_model(model, optimizer_ft, crit, exp_lr_scheduler, num_epochs=70)