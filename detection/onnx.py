import onnxruntime
import torch.onnx
from PIL import Image
import onnx
from model import Models
from dataset import CreatingDataloaderTrainVal


def ExportToONNX():
    dataloader = CreatingDataloaderTrainVal()
    model = Models()
    model = model.LoadModel()
    model.eval()
    x = next(iter(dataloader['val']))
    torch.onnx.export(model,
                (x,),
                "model3.onnx",
                #   export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names = ['input'],
                output_names = ['output'],
                dynamic_axes={'input' : {0 : 'batch_size'},
                                'output' : {0 : 'batch_size'}})

if __name__ == "__main__":
    ExportToONNX()