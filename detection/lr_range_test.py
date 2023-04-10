# !python lr_range.py
from torch_lr_finder import LRFinder
from model import Models
from dataset import CreatingDataloaderTrainVal

dataloader = CreatingDataloaderTrainVal()
model = Models()
model_test, optimizer, criterion = model.CreatingModel()

lr_finder = LRFinder(model_test, optimizer, criterion, device="cpu")
lr_finder.range_test(dataloader['val'], end_lr=1, num_iter=100)
lr_finder.plot()
plt.savefig('foo.png')
lr_finder.reset()