import numpy as np
import torch as th
from matplotlib import pyplot as plt
import cv2
from dataset import CreateDateloaderTest

device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')

BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
   
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 1)    
    cv2.rectangle(img, (x_min, y_min + int(1.0 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min , y_min + int(0.9 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1.0, 
        color=TEXT_COLOR, 
        lineType=cv2.LINE_AA,
    )
    return img


category_id_to_name = {0.0: '0', 1.0: '1'}

def visualize(image, bboxes, category_ids, name):
    category_id_to_name = {0.0: '0', 1.0: '1'}
    img = image
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(figsize=(9, 9))
    plt.axis('off')
    plt.savefig(f'{name}.png')
    plt.imshow(img)

import tqdm

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

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def draw_oll():
    dataloader = CreateDateloaderTest()
    for phase in ['train', 'val']:
        for img in dataloader[phase]:
            img = img.to(device)
            target = pars_txt()
            open('/content/detection/detection_db/1.txt', 'w').close()
            for i in range(len(img)):
                image = to_numpy(img[i].permute(1,2,0)).copy()
                targ = target[i]
                lbl = targ['labels']
                box = targ['boxes']
                for j in range(len(box)):
                    x_min, y_min, x_max, y_max = int(box[j][0]), int(box[j][1]), int(box[j][2]), int(box[j][3])
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=2)
                plt.figure(figsize=(9, 9))
                plt.axis('off')
                plt.imshow(image)
                plt.savefig(f'result/{i}_lbl.png')


def draw_result(offset, img, target, predict, predict_onnx):
    for i in range(len(img)):
        image = to_numpy(img[i].permute(1,2,0)).copy()
        targ = target[i]
        lbl = targ['labels']
        box = targ['boxes']
        for j in range(len(box)):
            x_min, y_min, x_max, y_max = int(box[j][0]), int(box[j][1]), int(box[j][2]), int(box[j][3])
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=2)
        plt.figure(figsize=(9, 9))
        plt.axis('off')
        plt.imshow(image)
        plt.savefig(f'result/{i}_lbl.png')

        targ = predict[i]
        lbl = targ['labels']
        box = targ['boxes']
        for j in range(len(box)):
            x_min, y_min, x_max, y_max = int(box[j][0]), int(box[j][1]), int(box[j][2]), int(box[j][3])
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=2)
        plt.figure(figsize=(9, 9))
        plt.axis('off')
        plt.imshow(image)
        plt.savefig(f'result/{i}_model.png')

        targ = predict_onnx[i + offset]
        lbl = targ['labels']
        box = targ['boxes']
        for j in range(len(box)):
            x_min, y_min, x_max, y_max = int(box[j][0]), int(box[j][1]), int(box[j][2]), int(box[j][3])
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=2)
        plt.figure(figsize=(9, 9))
        plt.axis('off')
        plt.imshow(image)
        plt.savefig(f'result/{i}_onnx.png')