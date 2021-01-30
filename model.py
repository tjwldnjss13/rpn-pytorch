import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.models as models
import matplotlib.pyplot as plt
from PIL import Image

from torchsummary import summary


class VGG(nn.Module):
    def __init__(self, cfgs, num_classes):
        super(VGG, self).__init__()
        self.cfgs = cfgs
        self.num_classes = num_classes
        self.conv_layers = make_conv_layers(cfgs)
        self.avg_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        # if self.num_classes >= 1000:
        #     self.classifier = nn.Sequential(
        #         nn.Linear(512 * 7 * 7, 4096),
        #         nn.ReLU(),
        #         nn.Linear(4096, 4096),
        #         nn.ReLU(),
        #         nn.Linear(4096, self.num_classes),
        #         nn.Softmax(),
        #     )
        # else:
        #     self.classifier = nn.Sequential(
        #         nn.Linear(512 * 7 * 7, 8192),
        #         nn.ReLU(),
        #         nn.Linear(8192, 4096),
        #         nn.ReLU(),
        #         nn.Linear(4096, 1024),
        #         nn.ReLU(),
        #         nn.Linear(1024, self.num_classes),
        #         nn.Softmax(),
        #     )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


def make_conv_layers(cfgs, batch_norm=True):
    layers = []
    prev_dim = 3
    conv1_flag = False

    for cfg in cfgs:
        if cfg == 'M':
            layers.append(nn.MaxPool2d(2, 2))
        elif cfg == 'conv1':
            conv1_flag = True
        else:
            if conv1_flag:
                layers.append(nn.Conv2d(prev_dim, cfg, 1, 1, 1))
                conv1_flag = False
            else:
                layers.append(nn.Conv2d(prev_dim, cfg, 3, 1, 1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(cfg))
            layers.append(nn.ReLU(True))
            prev_dim = cfg

    return nn.Sequential(*layers)


vgg_cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'C': [64, 64, 'M', 128, 128, 'M', 256, 256, 'conv1', 256, 'M', 512, 512, 'conv1', 512, 'M', 512, 512, 'conv1', 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

if __name__ == '__main__':
    vgg = VGG(vgg_cfgs['D'], 150).cuda()
    summary(vgg, (3, 224, 224))
