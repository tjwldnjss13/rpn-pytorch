import torch

from torchvision.models.detection import rpn, FasterRCNN

a = torch.Tensor([[1, 2, 3],
                  [4, 5, 6]])
b = (a % 2 == 0).nonzero()
print(a[b[:, 0], b[:, 1]])
