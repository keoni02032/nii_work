# !python lr_range.py
from torch_lr_finder import LRFinder
from model import Models
from dataset import CreatingDataloaderTrainVal

dataloader_train = CreatingDataloaderTrainVal()
model = Models()
model, criterion, optimizer_ft = model.CreateFTModel()

lr_finder = LRFinder(model, optimizer_ft, criterion, device="cpu")
lr_finder.range_test(dataloader_train['val'], end_lr=1, num_iter=100)
lr_finder.plot()
lr_finder.reset()