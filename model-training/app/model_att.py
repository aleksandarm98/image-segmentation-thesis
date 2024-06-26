# Attention UNET
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.psi(g1 + x1)
        return x * psi


class AttentionUNet(nn.Module):
    def __init__(self, hidden_size, in_channels=1, num_classes=1):
        super(AttentionUNet, self).__init__()
        # Downsampling path
        self.enc1 = DoubleConv(in_channels, hidden_size)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(hidden_size, hidden_size*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(hidden_size*2, hidden_size*4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(hidden_size*4, hidden_size*8)

        # Upsampling path
        self.up4 = nn.ConvTranspose2d(hidden_size * 8, hidden_size * 4, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(F_g=hidden_size * 4, F_l=hidden_size * 4, F_int=hidden_size * 2)
        self.dec4 = DoubleConv(hidden_size * 8, hidden_size * 4)
        self.up3 = nn.ConvTranspose2d(hidden_size * 4, hidden_size * 2, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=hidden_size * 2, F_l=hidden_size * 2, F_int=hidden_size)
        self.dec3 = DoubleConv(hidden_size * 4, hidden_size * 2)
        self.up2 = nn.ConvTranspose2d(hidden_size * 2, hidden_size, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=hidden_size, F_l=hidden_size, F_int=int(hidden_size / 2))
        self.dec2 = DoubleConv(hidden_size * 2, hidden_size)

        # Output layer
        self.final_conv = nn.Conv2d(hidden_size, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder part
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Decoder part with attention
        d4 = self.dec4(torch.cat([self.att4(self.up4(e4), e3), self.up4(e4)], dim=1))
        d3 = self.dec3(torch.cat([self.att3(self.up3(d4), e2), self.up3(d4)], dim=1))
        d2 = self.dec2(torch.cat([self.att2(self.up2(d3), e1), self.up2(d3)], dim=1))

        # Final output
        out = self.final_conv(d2)
        return torch.sigmoid(out)
