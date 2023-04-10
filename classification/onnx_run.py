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
import onnxruntime
import torch.onnx
from PIL import Image
import onnx
from dataset import CreatingDataloaderTrainVal, CreatingDataloaderTest
import numpy as np
device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

global running_corrects_onnx

def run(model):
    with th.no_grad():
        model.eval()
        running_corrects = 0
        running_corrects_onnx = 0
        ort_session = onnxruntime.InferenceSession("model1.onnx")
        for inputs, labels in tqdm(dataloader['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            predict = model(inputs)
            torch_out = predict
            ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(inputs)}
            ort_outs = ort_session.run(None, ort_inputs)
            _, preds = th.max(predict, 1)
            _, preds_onnx = th.max(th.tensor(ort_outs[0]).squeeze(0), 1)
            running_corrects += th.sum(preds == labels.data)
            running_corrects_onnx += th.sum(preds_onnx == labels)
        epoch_acc = running_corrects.double() / dataset_sizes_test['test']
        epoch_acc_onnx = running_corrects_onnx.double() / dataset_sizes_test['test']
    return epoch_acc, epoch_acc_onnx

if __name__ == "__main__":
    dataloader, dataset_sizes_test = CreatingDataloaderTest()
    model = Models()
    model = model.LoadTimmModel()
    epoch_acc, epoch_acc_onnx = run(model)
    print(epoch_acc)
    print(epoch_acc_onnx)