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

import time
def model_test(model):
    running_loss = 0.0
    running_corrects = 0
    model.eval()
    count = 0
    with th.no_grad():
        for inputs, labels in tqdm(dataloader_test['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            start = time.time()
            outputs = model(inputs)
            count += time.time() - start
            _, preds = th.max(outputs, 1)
            running_corrects += th.sum(preds == labels.data)
    epoch_acc = running_corrects.double() / dataset_sizes_test['test']
    print(epoch_acc)
    print(count / len(dataloader_test['test']))

if __name__ == "__main__":
    dataloader_test, dataset_sizes_test = CreatingDataloaderTest()
    model = Models()
    model = model.LoadTimmModel()
    model_test(model)