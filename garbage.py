import torch

from torchvision.models.detection import rpn, FasterRCNN

a = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = torch.Tensor([[1], [2], [3]])
print(a.shape)
print(b.shape)
