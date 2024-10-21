import os.path
import warnings

import torch
from torch import nn
from pvtv2 import *
import torch.nn.functional as F
from SC import *
from dysample import *
import torch


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Encoder(nn.Module):
    def __init__(self, pretrained_path=None, dim=16, num_heads=8):
        super(Encoder, self).__init__()
        self.backbone = pvt_v2_b2()
        if pretrained_path is None:
            warnings.warn('please provide the pretrained pvt model. Not using pretrained model.')
        elif not os.path.isfile(pretrained_path):
            warnings.warn(f'path: {pretrained_path} does not exists. Not using pretrained model.')
        else:
            print(f"using pretrained file: {pretrained_path}")
            save_model = torch.load(pretrained_path)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}# 匹配预训练模型的参数
            model_dict.update(state_dict)
            self.backbone.load_state_dict(model_dict)

    def forward(self, x):
        f1, f2, f3, f4 = self.backbone(x)
        return f1, f2, f3, f4



class MSF(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.convs = nn.ModuleList(
            [nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1) for _ in range(4)])
        self.dys1 = DySample(channel, scale=2, style='lp')
        self.dys2 = DySample(channel, scale=4, style='lp')
        self.dys3 = DySample(channel, scale=8, style='lp')

    def forward(self, xs, anchor):
        ans = torch.ones_like(anchor)
        target_size = anchor.shape[-1]

        for i, x in enumerate(xs):
            if x.shape[-1] > target_size:
                x = F.adaptive_avg_pool2d(x, (target_size, target_size))
            elif x.shape[-1] < target_size:
                if x.shape[-1] == target_size // 2:
                    x = self.dys1(x)
                elif x.shape[-1] == target_size // 4:
                    x = self.dys2(x)
                elif x.shape[-1] == target_size // 8:
                    x = self.dys3(x)
                                
            ans = ans * self.convs[i](x)

        return ans



class CMNet(nn.Module):

    def __init__(self, channel=32, n_classes=1, deep_supervision=True, pretrained_path=False):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = Encoder(pretrained_path)

        self.sc_1 = SC(64,64)
        self.sc_2 = SC(128,128)
        self.sc_3 = SC(320,320)
        self.sc_4 = SC(512,512)

        self.Translayer_1 = BasicConv2d(64, channel, 1)
        self.Translayer_2 = BasicConv2d(128, channel, 1)
        self.Translayer_3 = BasicConv2d(320, channel, 1)
        self.Translayer_4 = BasicConv2d(512, channel, 1)

        self.MSF_1 = MSF(channel)
        self.MSF_2 = MSF(channel)
        self.MSF_3 = MSF(channel)
        self.MSF_4 = MSF(channel)

        self.dysample1 = DySample(in_channels=channel, scale=2, style='lp')
        self.dysample2 = DySample(in_channels=channel, scale=2, style='lp')
        self.dysample3 = DySample(in_channels=channel, scale=2, style='lp')
        self.dysample4 = DySample(in_channels=channel, scale=2, style='lp')
        
        

        self.seg_outs = nn.ModuleList([
            nn.Conv2d(channel, n_classes, 1, 1) for _ in range(4)])


    def forward(self, x):
        seg_outs = []
        f1, f2, f3, f4 = self.encoder(x)

        f1 = self.sc_1(f1) * f1
        f2 = self.sc_2(f2) * f2
        f3 = self.sc_3(f3) * f3
        f4 = self.sc_4(f4) * f4
        
        
        f1 = self.Translayer_1(f1)
        f2 = self.Translayer_2(f2)
        f3 = self.Translayer_3(f3)
        f4 = self.Translayer_4(f4)


        f41 = self.MSF_4([f1, f2, f3, f4], f4)
        f31 = self.MSF_3([f1, f2, f3, f4], f3)
        f21 = self.MSF_2([f1, f2, f3, f4], f2)
        f11 = self.MSF_1([f1, f2, f3, f4], f1)


        y = self.dysample1(f41) + f31
        y = self.dysample2(y) + f21

        if self.deep_supervision:
            out1 = self.seg_outs[0](y).clone()

        y = self.dysample3(y) + f11
        out2 = self.seg_outs[1](y).clone()


        
        if self.deep_supervision:
            return F.interpolate(out1, scale_factor=8, mode='bilinear'), \
                F.interpolate(out2, scale_factor=4, mode='bilinear')
        else:
            return F.interpolate(out2, scale_factor=4, mode='bilinear')

if __name__ == "__main__":
    pretrained_path = "/code/U-Net_v2/model_pth/best_epoch_97.pth"
    model = CMNet(n_classes=1, deep_supervision=True, pretrained_path=pretrained_path)
    x = torch.rand((2, 3, 256, 256))
    ys = model(x)
    for y in ys:
        print(y.shape)
