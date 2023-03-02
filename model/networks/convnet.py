import torch
import torch.nn as nn


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_block, self).__init__()
        self.convlayer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.convlayer(x)


class conv_block_aff(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_block_aff, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, ii):
        x, r, b = ii[0], ii[1], ii[2]
        r, b = r.view(1, -1, 1, 1), b.view(1, -1, 1, 1)
        out = self.bn(self.conv(x))
        out = out * r + b
        out = self.pool(self.relu(out))
        return out


class ConvNetFront(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64):
        super(ConvNetFront, self).__init__()
        self.enc = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim)
        )

    def forward(self, x):
        return self.enc(x)


class ConvNetBack(nn.Module):
    def __init__(self, hid_dim=64, z_dim=64):
        super(ConvNetBack, self).__init__()
        self.encf1 = conv_block_aff(hid_dim, hid_dim)
        self.encf2 = conv_block_aff(hid_dim, z_dim)
        self.pool = nn.MaxPool2d(5)

    def forward(self, x, r, b):
        r1, b1, r2, b2 = r[:64], b[:64], r[64:], b[64:]
        x = self.encf1([x, r1, b1])
        x = self.encf2([x, r2, b2])
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x


def ConvNet():
    conv1, conv2 = ConvNetFront(), ConvNetBack()
    return conv1, conv2
