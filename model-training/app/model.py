import torch
import torch.nn as nn

from config import Config


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.config = Config()
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout)
        )

    def forward(self, x):
        return self.conv(x)


class UNetPlusPlus(nn.Module):
    def __init__(self, hidden_size, in_channels=1, num_classes=1):
        super(UNetPlusPlus, self).__init__()
        self.dc1 = DoubleConv(in_channels, hidden_size)
        self.dc2 = DoubleConv(hidden_size, hidden_size*2)
        self.dc3 = DoubleConv(hidden_size*2, hidden_size*4)
        self.dc4 = DoubleConv(hidden_size*4, hidden_size*8)
        self.dc5 = DoubleConv(hidden_size*8, hidden_size*16)

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dc12 = DoubleConv(hidden_size + hidden_size*2, hidden_size)
        self.dc23 = DoubleConv(hidden_size*2 + hidden_size*4, hidden_size*2)
        self.dc34 = DoubleConv(hidden_size*4 + hidden_size*8, hidden_size*4)
        self.dc45 = DoubleConv(hidden_size*8 + hidden_size*16, hidden_size*8)

        self.dc_up4 = DoubleConv(hidden_size*16 + hidden_size*8 + hidden_size*8, hidden_size*8)
        self.dc_up3 = DoubleConv(hidden_size*8 + hidden_size*4 + hidden_size*4, hidden_size*4)
        self.dc_up2 = DoubleConv(hidden_size*4 + hidden_size*2 + hidden_size*2, hidden_size*2)
        self.dc_up1 = DoubleConv(hidden_size*2 + hidden_size + hidden_size, hidden_size)

        self.final_conv = nn.Conv2d(hidden_size, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.dc1(x)
        p1 = self.pool(x1)
        x2 = self.dc2(p1)
        p2 = self.pool(x2)
        x3 = self.dc3(p2)
        p3 = self.pool(x3)
        x4 = self.dc4(p3)
        p4 = self.pool(x4)
        x5 = self.dc5(p4)

        x12 = self.dc12(torch.cat([x1, self.upsample(x2)], dim=1))
        x23 = self.dc23(torch.cat([x2, self.upsample(x3)], dim=1))
        x34 = self.dc34(torch.cat([x3, self.upsample(x4)], dim=1))
        x45 = self.dc45(torch.cat([x4, self.upsample(x5)], dim=1))

        x_up4 = self.upsample(x5)
        x_cat4 = torch.cat([x_up4, x4, x45], dim=1)
        x_up4 = self.dc_up4(x_cat4)

        x_up3 = self.upsample(x_up4)
        x_cat3 = torch.cat([x_up3, x3, x34], dim=1)
        x_up3 = self.dc_up3(x_cat3)

        x_up2 = self.upsample(x_up3)
        x_cat2 = torch.cat([x_up2, x2, x23], dim=1)
        x_up2 = self.dc_up2(x_cat2)

        x_up1 = self.upsample(x_up2)
        x_cat1 = torch.cat([x_up1, x1, x12], dim=1)
        x_up1 = self.dc_up1(x_cat1)

        out = self.final_conv(x_up1)
        return torch.sigmoid(out)
