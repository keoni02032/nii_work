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
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from vis import plt_result
import cv2
from matplotlib import pyplot as plt

device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def pars_txt():
    data = open('/content/detection/detection_db/1.txt')
    targets = []
    l = []
    d = {}
    while True:
        tmp = data.readline()
        if not tmp:
            break
        tmp = tmp.split(' ')
        if tmp == ['\n']:
            for i in range(len(l)):
                l[i].remove('\n')
                for j in range(len(l[i])):
                    l[i][j] = int(float(l[i][j]))
            tmp = th.tensor(l.pop()).long()
            if len(l) == 2:
                d['boxes'] = th.tensor(l)
            else:
                d['boxes'] = th.tensor(l)
            d['labels'] = tmp
            targets.append(d)
            l = []
            d = {}
        else:
            l.append(tmp)
    return targets

def run(model):
    pred_model = []
    targ_model = []
    pred = []
    with th.no_grad():
        model.eval()
        ort_session = onnxruntime.InferenceSession("model3.onnx")
        for img in tqdm(dataloader['val']):
            img = img.to(device)
            target = pars_txt()
            open('/content/detection/detection_db/1.txt', 'w').close()
            predict = model(img)
            torch_out = predict
            ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
            ort_outs = ort_session.run(None, ort_inputs)
            for i in range(len(predict)):
                pred_model.append(predict[i])
                targ_model.append(target[i])
                d = {}
                d['boxes'] = th.tensor(ort_outs[i * 3])
                d['scores'] = th.tensor(ort_outs[i * 3 + 1])
                d['labels'] = th.tensor(ort_outs[i * 3 + 2])
                pred.append(d)
        metric = MeanAveragePrecision()
        metric.update(pred_model, targ_model)
        map_50 = metric.compute()['map_50']
        print(map_50)
        map = metric.compute()['map']
        print(map)
        metric_onnx = MeanAveragePrecision()
        metric_onnx.update(pred, targ_model)
        map_50 = metric_onnx.compute()['map_50']
        print(map_50)
        map = metric_onnx.compute()['map']
        print(map)
        pred_model = []
        # pred = []
        targ_model = []
        # plt_result(img, predict, d, target)
        for i in range(len(img)):
            image = to_numpy(img[i].permute(1,2,0)).copy()
            targ = target[i]
            lbl = targ['labels']
            box = targ['boxes']
            print(image.shape)
            for j in range(len(box)):
                x_min, y_min, x_max, y_max = int(box[j][0]), int(box[j][1]), int(box[j][2]), int(box[j][3])
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=2)
            plt.figure(figsize=(9, 9))
            plt.axis('off')
            plt.imshow(image)
            plt.savefig(f'{i}_lbl.png')
            predi = predict[i]
            lbl = predi['labels']
            box = predi['boxes']
            print(image.shape)
            for j in range(len(box)):
                x_min, y_min, x_max, y_max = int(box[j][0]), int(box[j][1]), int(box[j][2]), int(box[j][3])
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=2)
            plt.figure(figsize=(9, 9))
            plt.axis('off')
            plt.imshow(image)
            plt.savefig(f'{i}_model.png')

            p = pred[i + 17]
            lbl = p['labels']
            box = p['boxes']
            print(image.shape)
            for j in range(len(box)):
                x_min, y_min, x_max, y_max = int(box[j][0]), int(box[j][1]), int(box[j][2]), int(box[j][3])
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=2)
            plt.figure(figsize=(9, 9))
            plt.axis('off')
            plt.imshow(image)
            plt.savefig(f'{i}_onnx.png')





if __name__ == "__main__":
    dataloader = CreatingDataloaderTrainVal()
    model = Models()
    model = model.LoadModel()
    run(model)