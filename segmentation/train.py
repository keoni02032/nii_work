import torch as th
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from torch.utils.tensorboard import SummaryWriter
# from model import Models
from dataset import CreatingDataloaderTrainVal
import tqdm
from tqdm import tqdm
from metrics import IoU, MIoU
import pandas as pd
from create_model import Models

device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')

CLASSES = [
    "vista",
    "building",
    "road",
    "tree",
    "cable",
]

def train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=40):
    r_losses = []
    val_losses = []
    rloss = 0
    a = ['ep'] + ['mode'] + CLASSES
    rm  = th.tensor([])
    next_desc = "first epoch"
    df_train = pd.DataFrame([], columns=a)
    df_val = pd.DataFrame([], columns=a)
    # writer = SummaryWriter()
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                for img,lbl in tqdm(dataloader[phase], desc = next_desc):
                    img = img.to(device)
                    lbl = lbl.to(device)
                    # print(img.shape)
                    with th.set_grad_enabled(phase == 'train'):
                        predict = model(img)#['out']
                        loss = criterion(predict, lbl)
                    m = IoU(lbl, predict)
                    rm = th.cat((rm, m))
                    loss.backward()
                    optimizer_ft.step()
                    optimizer_ft.zero_grad()
                    rloss += loss.detach().cpu().item() / len(dataloader)
                miou, class_arr = MIoU(rm)
                rm  = th.tensor([])
                df_train.loc[epoch] = [epoch] + [phase] + class_arr
                df_train.to_csv('./train_log.csv')
                r_losses += [rloss]
                exp_lr_scheduler.step()
                # writer.add_scalar('train loss', rloss, epoch)
                # writer.add_scalar('train MIoU', miou, epoch)
            else:
                model.eval()
                with th.no_grad():
                    for img,lbl in tqdm(dataloader[phase], desc = next_desc):
                        img = img.to(device)
                        lbl = lbl.to(device)
                        predict = model(img)#['out']
                        m = IoU(lbl, predict)
                        rm = th.cat((rm, m))
                        loss = criterion(predict, lbl)
                        rloss += loss.detach().cpu().item() / len(dataloader)
                    miou, class_arr = MIoU(rm)
                    rm  = th.tensor([])
                    df_val.loc[epoch] = [epoch+1] + [phase] + class_arr
                    df_val.to_csv('./val_log.csv')
                    val_losses += [rloss]
                    # writer.add_scalar('valtrain loss', rloss, epoch)
                    # writer.add_scalar('val MIoU', miou, epoch)
                    if epoch % 10 == 0:
                        th.save(model.state_dict(), f"./model_{epoch + 1}.pth")
            next_desc = f"Epoch of {phase}: [{epoch+1}], previous rloss: [{rloss:.3f}], MIoU: [{miou}]"
            if epoch == num_epochs - 1 and phase == 'val':
                print(f"Epoch of {phase}: [{epoch+1}], previous rloss: [{rloss:.3f}], MIoU: [{miou}]")
            rloss = 0
    # writer.close()

if __name__ == "__main__":
    dataloader = CreatingDataloaderTrainVal()
    model = Models()
    model, criterion, optimizer_ft, exp_lr_scheduler = model.CreatingModel()
    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=50)