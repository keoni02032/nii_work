import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch as th

device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')

CLASSES = [
    "vista",
    "building",
    "road",
    "tree",
    "cable",
]

def plt_result(img, predict, ort_outs, lbl):
    fig,axes = plt.subplots(len(lbl), 4, figsize = (12, len(lbl) * 4))
    for idc, (simg, spred, onnx_pred, slbl) in enumerate(zip(img.detach().cpu(), predict.detach().cpu().softmax(1), ort_outs.detach().cpu().squeeze().softmax(1), lbl.detach().cpu())):
        axes[idc, 0].imshow(simg.permute(1,2,0))
        axes[idc, 1].imshow(spred.argmax(0), vmin = 0, vmax = len(CLASSES)-1)
        axes[idc, 2].imshow(onnx_pred.argmax(0), vmin = 0, vmax = len(CLASSES)-1)
        axes[idc, 3].imshow(slbl.argmax(0), vmin = 0, vmax = len(CLASSES)-1)
    [ax.get_xaxis().set_visible(False) for ax in  fig.axes]
    [ax.get_yaxis().set_visible(False) for ax in fig.axes]
    plt.savefig('foo.png')
    fig.show()