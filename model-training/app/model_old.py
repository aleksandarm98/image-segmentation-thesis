import torch
import torch.nn as nn


################### smanjena arhitektura bez dense layera, pocetna arhitektura unetpp arhitekture
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)  # Dodavanje dropout sloja
        )

    def forward(self, x):
        return self.conv(x)


class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super(UNetPlusPlus, self).__init__()
        # Povećanje broja kanala u slojevima
        self.dc1 = DoubleConv(1, 32)
        self.dc2 = DoubleConv(32, 64)
        self.dc3 = DoubleConv(64, 128)
        self.dc4 = DoubleConv(128, 256)
        self.dc5 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Dodavanje dodatnih slojeva
        self.dc_up4 = DoubleConv(512 + 256, 256)
        self.dc_up3 = DoubleConv(256 + 128, 128)
        self.dc_up2 = DoubleConv(128 + 64, 64)
        self.dc_up1 = DoubleConv(64 + 32, 32)

        self.final_conv = nn.Conv2d(32, num_classes, 1)

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

        print(x1.shape)
        print(x2.shape)

        x_up4 = self.upsample(x5)
        x_cat4 = torch.cat([x_up4, x4], dim=1)
        x_up3 = self.dc_up4(x_cat4)

        x_up3 = self.upsample(x_up3)
        x_cat3 = torch.cat([x_up3, x3], dim=1)
        x_up2 = self.dc_up3(x_cat3)

        x_up2 = self.upsample(x_up2)
        x_cat2 = torch.cat([x_up2, x2], dim=1)
        x_up1 = self.dc_up2(x_cat2)

        x_up1 = self.upsample(x_up1)
        x_cat1 = torch.cat([x_up1, x1], dim=1)
        x_final = self.dc_up1(x_cat1)

        out = self.final_conv(x_final)
        return torch.sigmoid(out)


# # Training initialization
# model = UNetPlusPluss()
# train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=num_workers)
# val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
# criterion = bce_dice_loss
# optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)
# train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device=device)
#
