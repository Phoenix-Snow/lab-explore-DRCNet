import torch
from torch import nn


class CBS(nn.Module):
    def __init__(self, circulate_times, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(CBS, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.circulate_num = circulate_times

    def forward(self, x):
        for i in range(self.circulate_num):
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
        return x


class C3K2(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, shortcut=True, groups=1, expansion=0.5):
        super(C3K2, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU(inplace=True)
        self.m = nn.Sequential(*[Bottleneck(hidden_channels, hidden_channels, shortcut, groups, expansion=1.0) for _ in range(num_blocks)])
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.m(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x


class SPFF(nn.Module):
    def __init__(self, in_channels, out_channels, k=5):
        super(SPFF, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, k, 1, k//2, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.conv3 = nn.Conv2d(in_channels, in_channels, k, 1, k//2, bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.conv4 = nn.Conv2d(in_channels, in_channels, k, 1, k//2, bias=False)
        self.bn4 = nn.BatchNorm2d(in_channels)
        self.conv5 = nn.Conv2d(4 * in_channels, out_channels, 1, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        x3 = self.conv3(x2)
        x3 = self.bn3(x3)
        x3 = self.relu(x3)
        x4 = self.conv4(x3)
        x4 = self.bn4(x4)
        x4 = self.relu(x4)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.conv5(x)
        x = self.bn5(x)
        return x


class C2PSA(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, shortcut=True, groups=1, expansion=0.5):
        super(C2PSA, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU(inplace=True)
        self.m = nn.Sequential(*[Bottleneck(hidden_channels, hidden_channels, shortcut, groups, expansion=1.0) for _ in range(num_blocks)])
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.psa = PSA(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.m(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.psa(x)
        return x


class Bottleneck(nn.Module):
    """
    特征压缩和扩展
    """
    def __init__(self, in_channels, out_channels, shortcut, groups, expansion):
        super(Bottleneck, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.conv3 = nn.Conv2d(hidden_channels, out_channels, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut = shortcut
        if self.shortcut and in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, bias=False)
            self.bn_shortcut = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.shortcut:
            if identity.shape != x.shape:
                identity = self.conv_shortcut(identity)
                identity = self.bn_shortcut(identity)
            x += identity
        return x


class PSA(nn.Module):
    """
    空间注意力机制
    """
    def __init__(self, channels):
        super(PSA, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        b, c, h, w = x.size()
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.view(b, c, -1)
        x = self.softmax(x)
        x = x.view(b, c, h, w)
        return x
