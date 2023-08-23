# Adapted from: https://github.com/milesial/Pytorch-UNet
#Changes involve 
# 1) adding an Nbase parameter which changes the UNet
# size without changing the topology and
# 2) Adding an inpuy Batch normalization layer
from unet_parts_Copy import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, f_c,bilinear=False, Nbase = 16):
        super(UNet, self).__init__()
        with open(f_c, 'r') as f:
            channels = f.readlines()
        channels = [int(c.strip()) for c in channels]
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inpBN  = nn.BatchNorm2d(n_channels)
        self.inc = (DoubleConv(n_channels, channels[0], channels[1]))
        self.down1 = (Down(channels[2], channels[3],channels[4],channels[5]))
        self.down2 = (Down(channels[6], channels[7],channels[8],channels[9]))
        self.down3 = (Down(channels[10], channels[11],channels[12],channels[13]))
        factor = 2 if bilinear else 1
        self.down4 = (Down(channels[14], channels[15],channels[16],channels[17]))
        self.up1 = (Up(channels[18], channels[19],channels[20], bilinear))
        self.up2 = (Up(channels[21], channels[22],channels[23], bilinear))
        self.up3 = (Up(channels[24], channels[25],channels[26], bilinear))
        self.up4 = (Up(channels[27], channels[28],channels[29], bilinear))
        self.outc = (OutConv(channels[29], n_classes))

    def forward(self, x):
        x1 = self.inc(self.inpBN(x))
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
