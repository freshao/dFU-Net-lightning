
import torch.nn as nn
import torch
from torch import autograd

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class ResidualPath(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(ResidualPath,self).__init__()
        self.resblock = nn.Sequential(
            nn.ConvTranspose2d(in_ch,out_ch,2,stride =2),
            nn.ReLU(inplace=True)
        )
    def forward(self,input):
        return self.resblock(input)


class PAM_Module(nn.Module):
    def __init__(self,in_dim):
        super(PAM_Module,self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_dim,in_dim//8,1)
        self.key_conv = nn.Conv2d(in_dim,in_dim//8,1)
        self.value_conv = nn.Conv2d(in_dim,in_dim,1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim = -1)

    def forward(self,x):
        m_batchsize,C,height,width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1)
        proj_key = self.key_conv(x).view(m_batchsize,-1,width*height)
        energy = torch.bmm(proj_query,proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height)

        out = torch.bmm(proj_value,attention.permute(0,2,1))
        out = out.view(m_batchsize,C,height,width)

        out = self.gamma*out + x
        return out


class CAM_Module(nn.Module):
    def __init__(self,in_dim):
        super(CAM_Module,self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim = -1)
    
    def forward(self,x):
        m_batchsize,C, height, width = x.size()
        proj_query = x.view(m_batchsize,C,-1)
        porj_key = x.view(m_batchsize,C,-1).permute(0,2,1)
        energy = torch.bmm(proj_query,porj_key)
        energy_new = torch.max(energy,-1,keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize,C,-1)

        out = torch.bmm(attention,proj_value)
        out = out.view(m_batchsize,C,height,width)
        
        out = self.gamma*out +x
        return out


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

class UMDnet(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(UMDnet, self).__init__()

        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.convpath1=Convres(128,64)
        self.upres1 = ResidualPath(64,64)

        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.convpath2=Convres(256,128)
        self.upres2 = ResidualPath(128,128)

        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.convpath3=Convres(512,256)
        self.upres3 = ResidualPath(256,256)

        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.convpath4=Convres(1024,512)
        self.upres4 = ResidualPath(512,512)

        self.conv5 = DoubleConv(512, 1024)
        self.pam = PAM_Module(1024)
        self.cam = CAM_Module(1024)

        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)

        self.up7 = nn.ConvTranspose2d(512,256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)

        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)

        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)

        self.conv10 = nn.Conv2d(64,out_ch, 1)
    
    def forward(self,x):
        c1=self.conv1(x)
        p1=self.pool1(c1)
        r1=self.upres1(p1)
        merge1 = torch.cat([c1,r1],dim=1)
        c11=self.convpath1(merge1)

        c2=self.conv2(p1)
        p2=self.pool2(c2)
        r2=self.upres2(p2)
        merge2 = torch.cat([c2,r2],dim=1)
        c22=self.convpath2(merge2)

        c3=self.conv3(p2)
        p3=self.pool3(c3)
        r3=self.upres3(p3)
        merge3 = torch.cat([c3,r3],dim=1)
        c33=self.convpath3(merge3)

        c4=self.conv4(p3)
        p4=self.pool4(c4)
        r4=self.upres4(p4)
        merge4 = torch.cat([c4,r4],dim=1)
        c44=self.convpath4(merge4)

        c5=self.conv5(p4)
        pa5 =self.pam(c5)
        ca5 =self.cam(c5)
        
        sum5 = pa5+ca5 

        up_6= self.up6(sum5)
        merge6 = torch.cat([up_6, c44], dim=1)
        c6=self.conv6(merge6)
        up_7=self.up7(c6)
        merge7 = torch.cat([up_7, c33], dim=1)
        c7=self.conv7(merge7)
        up_8=self.up8(c7)
        merge8 = torch.cat([up_8, c22], dim=1)
        c8=self.conv8(merge8)
        up_9=self.up9(c8)
        merge9=torch.cat([up_9,c11],dim=1)
        c9=self.conv9(merge9)
        c10=self.conv10(c9)
        out = nn.Sigmoid()(c10)
        return out

