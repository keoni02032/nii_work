import torch as th
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from torch.utils.tensorboard import SummaryWriter
from model import Models
from dataset import CreatingDataloaderTrainVal
import tqdm
from tqdm import tqdm
import onnxruntime
import torch.onnx
from PIL import Image
import onnx
from metrics import IoU, MIoU
from vis import plt_result

device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def run(model):
    rm  = th.tensor([])
    rm_2 = th.tensor([])

    with th.no_grad():
        model.eval()
        ort_session = onnxruntime.InferenceSession("model2.onnx")
        for img,lbl in tqdm(dataloader['val']):
            img = img.to(device)
            lbl = lbl.to(device)
            predict = model(img)['out']
            torch_out = predict
            ort_outs = th.tensor(ort_session.run(None, {'input': to_numpy(img)})).to(device)
            m = IoU(lbl, predict)
            m_2 = IoU(lbl, ort_outs[0])
            rm = th.cat((rm, m))
            rm_2 = th.cat((rm, m))        
        miou, class_arr = MIoU(rm)
        miou_onnx, cl = MIoU(rm_2)
        rm  = th.tensor([])
        rm_2 = th.tensor([])
        to_numpy(img)
        to_numpy(lbl)
        to_numpy(ort_outs)
        to_numpy(predict)
        print(miou)
        print(miou_onnx)
        print(miou_onnx - miou)
        plt_result(img, predict, ort_outs, lbl)

if __name__ == "__main__":
    dataloader = CreatingDataloaderTrainVal()
    model = Models()
    model = model.LoadModel()
    run(model)