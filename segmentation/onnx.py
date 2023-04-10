import onnxruntime
import torch.onnx
from PIL import Image
import onnx
from model import Models
from dataset import CreatingDataloaderTrainVal
import torch as th

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')

def ExportToONNX():
    dataloader = CreatingDataloaderTrainVal()
    model = Models()
    model = model.LoadModel()
    model.eval()
    x = next(iter(dataloader['val']))
    torch.onnx.export(model,
                (x[0],),
                "model2.onnx",
                export_params=True,
                opset_version=16,
                do_constant_folding=False,
                input_names = ['input'],
                output_names = ['output'],)
                # dynamic_axes={'input' : {0 : 'batch_size'},
                #               'output' : {0 : 'batch_size'}})

if __name__ == "__main__":
    ExportToONNX()