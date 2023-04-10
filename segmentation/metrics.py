import torch as th
import numpy as np

def IoU(target, predicted):
    result = th.tensor([])
    for i in range(target.shape[0]):
        iousum = []
        target_array = target[i, :, :, :].argmax(0)
        predicted_array = predicted[i, :, :, :].argmax(0)
        for j in range(5):
            target_arr = target_array.eq(j)
            predicted_arr = predicted_array.eq(j)
            intersection = th.logical_and(target_arr, predicted_arr).sum()
            union = th.logical_or(target_arr, predicted_arr).sum()
            if union == 0:
                iou_score = -1.0
            elif intersection == 0:
                iou_score = 0.0
            else:
                iou_score = intersection / union
            iousum.append(iou_score)
        result = th.cat((result, th.tensor([iousum])))
    return result



def MIoU(iou):
    d = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0}
    count = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0}
    for i in range(len(iou)):
        for j in range(len(iou[i])):
            if iou[i][j] >= 0:
                count[str(j)] += 1
                if iou[i][j] > 0:
                    d[str(j)] += iou[i][j]
    val = 0
    result = []
    for i in range(5):
        result.append(float(d[str(i)] / count[str(i)]))
        val += d[str(i)] / count[str(i)]
    val /= 5
    return val, result