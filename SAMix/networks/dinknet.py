"""
Codes of LinkNet based on https://github.com/snakers4/spacenet-three
"""
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

from functools import partial

nonlinearity = partial(F.relu, inplace=True)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class _ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(_ASPP, self).__init__()
        out_channels = in_channels
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b1 = _ASPPConv(in_channels, out_channels, rate1)
        self.b2 = _ASPPConv(in_channels, out_channels, rate2)
        self.b3 = _ASPPConv(in_channels, out_channels, rate3)
        self.b4 = _AsppPooling(in_channels, out_channels)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat3 = self.b2(x)
        feat4 = self.b3(x)
        feat5 = self.b4(x)
        x = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        x = self.project(x)
        return x

class _ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rate):
        super(_ASPPConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate, dilation=atrous_rate, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class _AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_AsppPooling, self).__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        size = x.size()[2:]
        pool = self.gap(x)
        out = F.interpolate(pool, size, mode='bilinear', align_corners=True)
        return out

class Dblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Dblock, self).__init__()
        self.dialte1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=1, padding=1)
        self.dialte2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=2, padding=2)
        self.dialte3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=4, padding=4)
        self.dialte4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=8, padding=8)
        #self.dialte5 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dialte1_out = nonlinearity(self.dialte1(x))
        dialte2_out = nonlinearity(self.dialte2(dialte1_out))
        dialte3_out = nonlinearity(self.dialte3(dialte2_out))
        dialte4_out = nonlinearity(self.dialte4(dialte3_out))
        #dialte5_out = nn.ReLU(self.dialte1(dialte4_out), inplace=True)
        out = x + dialte1_out + dialte2_out + dialte3_out + dialte4_out #+ dialte5_out
        return out


class DinkNet34_new(nn.Module):
    def __init__(self, num_classes=1):
        super(DinkNet34_new, self).__init__()
        filters = [64, 128, 256, 512]
        ResNet = models.resnet34(pretrained=True)

        self.firstconv = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.firstbn = ResNet.bn1
        self.firstrelu = ResNet.relu
        self.firstmaxpool = ResNet.maxpool
        self.encoder1 = ResNet.layer1
        self.encoder2 = ResNet.layer2
        self.encoder3 = ResNet.layer3
        self.encoder4 = ResNet.layer4

        self.dblock = Dblock(512,512)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)


    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return torch.sigmoid(out)


pretrained_mean = torch.tensor([0.485, 0.456, 0.406], requires_grad = False).view((1, 3, 1, 1))
pretrained_std = torch.tensor([0.229, 0.224, 0.225], requires_grad = False).view((1, 3, 1, 1))

