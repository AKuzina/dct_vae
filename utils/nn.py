import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.thirdparty.blurpool import BlurPool

class Siren(nn.Module):
    def __init__(self):
        super(Siren, self).__init__()
        self.w_0 = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        return torch.sin(self.w_0 * x)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class ConvBlock(nn.Module):
    """
    if forward true:
        BatchNorm (if needed) + Activation + Convolution
    else:
        Convolution + BatchNorm (if needed) + Activation
    """
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 act=nn.ReLU(), weight_norm=False, batch_norm=False, forward=True):
        super(ConvBlock, self).__init__()
        conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=1, padding=padding,
                         dilation=dilation, groups=groups)

        if weight_norm:
            conv = nn.utils.weight_norm(conv)
        net = []
        if forward:
            if batch_norm:
                net.append(nn.BatchNorm2d(in_ch, momentum=0.05))
            if act is not None:
                net.append(act)
            net += [conv]
            if stride == 2:
                net += [nn.AvgPool2d(kernel_size=2, stride=2)]
        else:
            net.append(conv)
            if stride == 2:
                net += [nn.AvgPool2d(kernel_size=2, stride=2)]
            if batch_norm:
                net.append(nn.BatchNorm2d(out_ch, momentum=0.05))
            if act is not None:
                net.append(act)
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class ConvTransposeBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, dilation=1,
                 output_padding=0, act=nn.ReLU(), weight_norm=False, batch_norm=False):
        super(ConvTransposeBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride=stride,
                                       padding=padding, dilation=dilation,
                                       output_padding=output_padding)
        self.activation = act
        self.bn = nn.BatchNorm2d(out_ch, momentum=0.05) if batch_norm else None
        if weight_norm:
            self.conv = nn.utils.weight_norm(self.conv)

    def forward(self, x):
        out = self.conv(x)
        if self.bn is not None:
            out = self.bn(out)
        if self.activation is not None:
            out = self.activation(out)
        return out


class _ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, use_res=True):
        super(_ResBlock, self).__init__()
        self.use_res = use_res
        self.beta = nn.Parameter(torch.tensor([0.]), requires_grad=True)

        if stride == 1:
            if in_channels == out_channels:
                self.skip = nn.Identity()
            else:
                self.skip = nn.Conv2d(in_channels, out_channels, 1)
        elif stride == 2:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
                nn.AvgPool2d(kernel_size=2, stride=2)
            )
        elif stride == -1:
            self.skip = nn.Sequential(nn.Upsample(scale_factor=2),
                                      nn.Conv2d(in_channels, out_channels, 1))
        self.net = None

    def forward(self, x):
        if self.use_res:
            return self.skip(x) + self.beta * self.net(x)
        else:
            return self.net(x)


class ConcatELU(nn.Module):
    """
    Activation function that applies ELU in both direction (inverted and plain).
    Allows non-linearity while providing strong gradients for any input (important for final convolution)
    """

    def forward(self, x):
        return torch.cat([F.elu(x), F.elu(-x)], dim=1)

class Linear1d(nn.Module):
    def __init__(self, num_channels):
        super(Linear1d, self).__init__()
        self.gain = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x):
        return x * self.gain + self.bias
