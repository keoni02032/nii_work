import onnxruntime
import torch.onnx
from PIL import Image
import onnx
from model import Models
from dataset import CreatingDataloaderTrainVal, CreatingDataloaderTest
import torch as th

device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def ExportToONNX():
    dataloader, dataset_sizes_test = CreatingDataloaderTest()
    model = Models()
    model = model.LoadTimmModel()
    model.eval()
    x = next(iter(dataloader['test']))
    print(x[0].shape)
    torch.onnx.export(model,
                x[0].to(device),
                "model1.onnx",
                export_params=True,
                opset_version=16,
                do_constant_folding=True,
                input_names = ['input'],
                output_names = ['output'],
                dynamic_axes={'input' : {0 : 'batch_size'},
                            'output' : {0 : 'batch_size'}})

if __name__ == "__main__":
    ExportToONNX()