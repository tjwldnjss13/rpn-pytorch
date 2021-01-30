import numpy as np
import torch
import torch.nn as nn

from utils import calculate_ious
from rpn_utils import anchor_box_generator


class RPN(nn.Module):
    def __init__(self, in_dim, out_dim, in_size, n_anchor):
        super(RPN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_size = in_size
        self.conv = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.conv.weight.data.normal_(0, .01)
        self.conv.bias.data.zero_()
        self.reg_layer = nn.Conv2d(out_dim, n_anchor * 4, 1, 1, 0)
        # self.reg_layer = nn.Linear(out_dim * self.in_size[0] * in_size[1], n_anchor * 4)
        self.reg_layer.weight.data.normal_(0, .01)
        self.reg_layer.bias.data.zero_()
        self.cls_layer = nn.Conv2d(out_dim, n_anchor * 2, 1, 1, 0)
        # self.cls_layer = nn.Linear(out_dim * self.in_size[0] * self.in_size[1], n_anchor * 2)
        self.cls_layer.weight.data.normal_(0, .01)
        self.cls_layer.bias.data.zero_()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        x = self.conv(x)

        reg = self.reg_layer(x)
        cls = self.cls_layer(x)

        # reg = reg.permute(0, 2, 3, 1).contiguous().view(reg.size(0), -1, 4)
        # cls = cls.permute(0, 2, 3, 1).contiguous().view(cls.size(0), -1, 2)

        # cls = self.softmax(cls)

        return reg, cls


if __name__ == '__main__':
    import cv2 as cv
    import matplotlib.pyplot as plt
    import copy

    rpn = RPN(512, 512, (14, 14), 9).cuda()
    from torchsummary import summary
    summary(rpn, (512, 14, 14))

    ratios = [.5, 1, 2]
    scales = [128, 256, 512]
    in_size = (600, 1000)
    anchor_boxes = anchor_box_generator(ratios, scales, in_size, 16)

    img_pth = 'sample/dogs.jpg'
    img = cv.imread(img_pth)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_h_og, img_w_og, _ = img.shape
    img = cv.resize(img, (in_size[1], in_size[0]))

    bbox = np.array([[120, 70, 570, 280], [220, 270, 580, 450], [30, 440, 570, 700]])
    bbox[:, 0] = bbox[:, 0] * (in_size[0] / img_h_og)
    bbox[:, 1] = bbox[:, 1] * (in_size[1] / img_w_og)
    bbox[:, 2] = bbox[:, 2] * (in_size[0] / img_h_og)
    bbox[:, 3] = bbox[:, 3] * (in_size[1] / img_w_og)

    img_copy = copy.deepcopy(img)

    for i, box in enumerate(anchor_boxes):
        y1, x1, y2, x2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv.rectangle(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)

    for i, gt in enumerate(bbox):
        y1, x1, y2, x2 = int(gt[0]), int(gt[1]), int(gt[2]), int(gt[3])
        cv.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

    plt.figure(figsize=(10, 6))
    plt.imshow(img_copy)
    plt.show()