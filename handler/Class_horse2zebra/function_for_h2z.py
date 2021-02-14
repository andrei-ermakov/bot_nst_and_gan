import functools
from torch.nn import init
import torch.nn as nn
import torch


# получить слой батч-нормализации
def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        # affine - True -> гамма и бэта обучаемые параметры
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        # Если track_running_stats установлено значение True, во время обучения этот слой продолжает
        # выполнять оценки своего вычисленного среднего значения и дисперсии, которые затем используются для нормализации во время оценки
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


"""
hasattr(obj, name) -> bool
Возвращает флаг, указывающий на то, содержит ли объект указанный атрибут.
obj : object Объект, существование атрибута в котором нужно проверить.
name : str Имя атрибута, существование которого требуется проверить.
"""


# инициализация весов:
# коэффициент - это значение из норм распр с мо=0, ско=0.02
def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            # init normal - заполняет m.weight.data значениями,
            # распределенными по нормальному закону с МО=0.0 и СКО=gain
            init.normal_(m.weight.data, 0.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal(m.weight.data, 1.0, gain)
            init.constant(m.bias.data, 0.0)

    net.apply(init_func)


# инициализация сети и перевод на GPU
def init_network(net, gpu_ids=list):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda(gpu_ids[0])
        # DataParallel -> Реализует параллелизм данных на уровне модуля.
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net)
    return net


def conv_norm_relu(in_dim, out_dim, kernel_size, stride=1, padding=0,
                   norm_layer=nn.BatchNorm2d, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=bias),
        norm_layer(out_dim), nn.ReLU(True))


def dconv_norm_relu(in_dim, out_dim, kernel_size, stride=1, padding=0, output_padding=0,
                    norm_layer=nn.BatchNorm2d, bias=False):
    return nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride,
                           padding, output_padding, bias=bias),
        norm_layer(out_dim), nn.ReLU(True))


class ResidualBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_dropout, use_bias):
        super(ResidualBlock, self).__init__()
        res_block = [nn.ReflectionPad2d(1),
                     conv_norm_relu(dim, dim, kernel_size=3,
                     norm_layer=norm_layer, bias=use_bias)]
        if use_dropout:
            res_block += [nn.Dropout(0.5)]
        res_block += [nn.ReflectionPad2d(1),
                      nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
                      norm_layer(dim)]

        self.res_block = nn.Sequential(*res_block)

    def forward(self, x):
        return x + self.res_block(x)


def set_grad(nets, requires_grad=False):
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad


# To load the checkpoint
def load_checkpoint(ckpt_path, map_location=None):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    return ckpt


# To make cuda tensor
def cuda(xs):
    if torch.cuda.is_available():
        if not isinstance(xs, (list, tuple)):
            return xs.to("cuda")
        else:
            return [x.cuda() for x in xs]