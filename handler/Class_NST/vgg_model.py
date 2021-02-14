import torch.nn as nn
from torchvision import models


def get_vgg():
    # загрузим модель
    vgg = models.vgg19(pretrained=True)

    # отключим градиенты
    for param in vgg.parameters():
        param.requires_grad = False

    for i, layer in enumerate(vgg.features):
        if isinstance(layer, nn.MaxPool2d):
            vgg.features[i] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    return vgg
