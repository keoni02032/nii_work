import torch as th
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from torch.utils.tensorboard import SummaryWriter
from model import Models
from dataset import CreatingDataloaderTrainVal, CreatingDataloaderTest
import tqdm
from tqdm import tqdm

device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    # writer = SummaryWriter()
    next_desc = "first epoch"
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
          running_loss = 0.0
          running_corrects = 0
          if phase == 'train':
              model.train()
              for inputs, labels in tqdm(dataloader_train[phase], desc = next_desc):
                  inputs = inputs.to(device)
                  labels = labels.to(device)
                  optimizer.zero_grad()
                  # with th.set_grad_enabled(phase == 'train'):
                  outputs = model(inputs)
                  _, preds = th.max(outputs, 1)
                  loss = criterion(outputs, labels)
                  loss.backward()
                  optimizer.step()
                  running_loss += loss.item() * inputs.size(0)
                  running_corrects += th.sum(preds == labels.data)
              scheduler.step()
              epoch_loss = running_loss / dataset_sizes_train[phase]
              epoch_acc = running_corrects.double() / dataset_sizes_train[phase]
              # writer.add_scalar('train loss', epoch_loss, epoch)
              # writer.add_scalar('train accuracy', epoch_acc, epoch)
          else:
              model.eval()
              with th.no_grad():
                  for inputs, labels in tqdm(dataloader_train[phase], desc = next_desc):
                      inputs = inputs.to(device)
                      labels = labels.to(device)
                      outputs = model(inputs)
                      _, preds = th.max(outputs, 1)
                      running_loss += loss.item() * inputs.size(0)
                      running_corrects += th.sum(preds == labels.data)
              epoch_loss = running_loss / dataset_sizes_train[phase]
              epoch_acc = running_corrects.double() / dataset_sizes_train[phase]
              if epoch % 10 == 0:
                  th.save(model.state_dict(), f"./model_{epoch}.pth")
                  th.save(optimizer.state_dict(), f"./optim_{epoch}.pth")
              # writer.add_scalar('validation loss', epoch_loss, epoch)
              # writer.add_scalar('validation accuracy', epoch_acc, epoch)
          next_desc = f"Epoch of {phase}: [{epoch+1}], loss: [{epoch_loss:.3f}], acc: [{epoch_acc}]"
          if epoch == num_epochs - 1 and phase == 'val':
              print(f"Epoch of {phase}: [{epoch+1}], loss: [{epoch_loss:.3f}], acc: [{epoch_acc}]")
    # writer.close()
    return model

if __name__ == "__main__":
    dataloader_train, dataset_sizes_train = CreatingDataloaderTrainVal()
    model = Models()
    model_ft, criterion, optimizer_ft, exp_lr_scheduler = model.CreatingTimmModel()
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=80)