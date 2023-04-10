import torch
from torch import nn

class InputConv(nn.Module):
    def __init__(self, inchanels, out):
        super(InputConv, self).__init__()
        self.inp_conv = nn.Sequential(
            nn.Conv2d(inchanels, out, kernel_size=3, padding=1),
            nn.BatchNorm2d(out),
            nn.ReLU(inplace=True),
            nn.Conv2d(out, out, kernel_size=3, padding=1),
            nn.BatchNorm2d(out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.inp_conv(x)

class UP(nn.Module):
    def __init__(self, inchanels, out):
        super(UP, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        self.inp_conv = InputConv(inchanels, out)
        

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.inp_conv(x1)

class Down(nn.Module):
    def __init__(self, inchanels, out):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            InputConv(inchanels, out)
        )

    def forward(self, x):
        return self.down(x)

class Out(nn.Module):
    def __init__(self, inchanels, out):
        super(Out, self).__init__()
        self.out = nn.Conv2d(inchanels, out, kernel_size=1)

    def forward(self, x):
        return self.out(x)

class UNet(nn.Module):
    def __init__(self, inchanels, classes):
        super(UNet, self).__init__()
        self.inchanels = inchanels
        self.classes = classes
        self.inp = InputConv(inchanels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = UP(1024, 512)
        self.up2 = UP(512, 256)
        self.up3 = UP(256, 128)
        self.up4 = UP(128, 64)
        self.out = Out(64, classes)

    def forward(self, x):
        x_1 = self.inp(x)
        x_2 = self.down1(x_1)
        x_3 = self.down2(x_2)
        x_4 = self.down3(x_3)
        x_5 = self.down4(x_4)
        x = self.up1(x_5, x_4)
        x = self.up2(x, x_3)
        x = self.up3(x, x_2)
        x = self.up4(x, x_1)
        result = self.out(x)
        return result