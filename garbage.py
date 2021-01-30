import torch

from torchvision.models.detection import rpn, FasterRCNN

a = torch.Tensor([1, 0, 0, 1, 0, 1]).type(torch.IntTensor)
b = torch.Tensor([1, 1, 0, 0, 0, 0]).type(torch.IntTensor)

and_ = torch.bitwise_and(a, b)
or_ = torch.bitwise_or(a, b)

print(and_.sum() / or_.sum())
