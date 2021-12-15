import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
# import pdb
# import matplotlib.pyplot as plt
# import random
# import torch.utils.model_zoo as model_zoo
from network.DenseUorigin import DenseUNet
from network.resnet import resnet34
from network.origin_ce import DACblock_with_inception
from network.swin_transformer import Swin_uper

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

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

nonlinearity = partial(F.relu, inplace=True)

class ResidualPath(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(ResidualPath,self).__init__()
        self.resblock = nn.Sequential(
            nn.ConvTranspose2d(in_ch,out_ch,2,stride =2),
            nn.ReLU(inplace=True)
        )
    def forward(self,input):
        return self.resblock(input)

class Convres(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Convres,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(4 * dim, 2 * dim, kernel_size=1, bias=False)
        # self.norm = norm_layer(4 * dim)
        self.norm = nn.BatchNorm2d(4 * dim)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, C, H, W = x.shape
        # assert h == H and w == W, "input feature has wrong size"

        # x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, :, 0::2, 0::2]  # B H/2 W/2 C
        x1 = x[:, :, 1::2, 0::2]  # B H/2 W/2 C
        x2 = x[:, :, 0::2, 1::2]  # B H/2 W/2 C
        x3 = x[:, :, 1::2, 1::2]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, 4 * C, H // 2, W // 2)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

class dFUNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=1, num_channels=3):
        super(dFUNet, self).__init__()
        self.n_classes = num_classes
        filters = [64, 128, 256, 512]
        resnet = resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # self.upers1 = ResidualPath(filters[0], filters[0])
        # self.pool1 = PatchMerging(filters[0])
        # self.upers1_1 = ResidualPath(filters[0] * 2, filters[0] * 2)
        # self.convpath1 = Convres(filters[0] * 4, filters[0])

        self.upers2 = ResidualPath(filters[1], filters[1])
        self.pool2 = PatchMerging(filters[0])
        self.upers2_1 = ResidualPath(filters[0] * 2, filters[0] * 2)
        self.convpath2 = Convres(filters[1] * 2 + filters[0], filters[0])

        self.upers3 = ResidualPath(filters[2], filters[2])
        self.pool3 = PatchMerging(filters[1])
        self.upers3_1 = ResidualPath(filters[1] * 2, filters[1] * 2)
        self.convpath3 = Convres(filters[2] * 2 + filters[1], filters[1])


        self.upers4 = ResidualPath(filters[3], filters[3])
        self.pool4 = PatchMerging(filters[2])
        self.upers4_1 = ResidualPath(filters[2] * 2, filters[2] * 2)
        self.convpath4 = Convres(filters[3] * 2 + filters[2], filters[2])

        self.dblock = DACblock_with_inception(512)

        self.decoder4 = DecoderBlock(512, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        # self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finaldeconv1 = nn.Conv2d(filters[0], 32, 3, padding=1)
        self.finalrelu1 = nonlinearity
        # self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        # self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)
        # self.inch = nn.Conv2d(in_ch, 3, 1)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.bn01 = nn.BatchNorm2d(3)
        # self.swin = Swin_uper()
        # self.finalconv4 = nn.Conv2d(num_classes * 2, num_classes, 3, padding=1)
        # self.finalrelu4 = nonlinearity

    def forward(self, x):
        # x, res_input = self.res_img(x)
        # swin_x, x1 = self.swin(res_input)
        # Encoder
        # x = self.inch(res_input)
        # x = self.relu1(x)
        # x = self.bn01(x)
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        # x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        # up1 = self.upers1(e1)
        # up11 = self.upers1_1(self.pool1(x))
        # merge1 = torch.cat([up1,up11,x], dim=1)
        # c11 = self.convpath1(merge1)

        e2 = self.encoder2(e1)
        up2 = self.upers2(e2)
        # p2 = self.pool2(e1)
        up22 = self.upers2_1(self.pool2(e1))
        merge2 = torch.cat([up2,up22,e1], dim=1)
        c22 = self.convpath2(merge2)

        e3 = self.encoder3(e2)
        up3 = self.upers3(e3)
        up33 = self.upers3_1(self.pool3(e2))
        merge3 = torch.cat([up3,up33,e2], dim=1)
        c33 = self.convpath3(merge3)

        e4 = self.encoder4(e3)

        up4 = self.upers4(e4)
        up44 = self.upers4_1(self.pool4(e3))
        merge4 = torch.cat([up4,up44,e3], dim=1)
        c44 = self.convpath4(merge4)

        # Center
        e4 = self.dblock(e4)
        # e4 = self.spp(e4)

        # Decoder
        d4 = self.decoder4(e4) + c44
        d3 = self.decoder3(d4) + c33
        d2 = self.decoder2(d3) + c22
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv3(out)


        return torch.sigmoid(out)


class FitNet(nn.Module):
    def __init__(self, model, in_ch=3, num_classes=1):
        super(FitNet, self).__init__()
        self.fit = model
        self.n_classes = num_classes
        self.inch = nn.Conv2d(in_ch, 3, 3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn01 = nn.BatchNorm2d(3)

    def forward(self, x):
        x = self.inch(x)
        x = self.bn01(x)
        x = self.relu1(x)
        x = self.fit(x)

        return x
        # return torch.sigmoid(x)

# TODO AxialNet
class AxialNet(nn.Module):
    def __int__(self):
        super(AxialNet, self).__int__()
        '''
        q is Res images Encoder by Axial row/col
        k is BottleNeck Encoder by 2D
        v is input image by Axial row/col
        
        position embedding
        
        '''
# TODO SmileNet

if __name__ == '__main__':
    # model = ghostnet()
    model = dFUNet(in_ch=3)
    model = FitNet(model, 4)
    # model = model.
    model.eval()
    print(model)
    input = torch.randn(1, 5, 512, 512)
    y = model(input)
    print(y.size())
