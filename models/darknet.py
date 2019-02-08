#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn

from modules import InPlaceABNSync as BatchNorm2d


class ConvBNLeaky(nn.Module):
    def __init__(self,
            in_chan,
            out_chan,
            ks = 3,
            stride = 1,
            pad = 0,
            dilation = 1,
            slope= 0.1,
            *args, **kwargs):
        super(ConvBNLeaky, self).__init__()

        self.conv = nn.Conv2d(
                in_chan,
                out_chan,
                kernel_size = ks,
                padding = pad,
                dilation = dilation,
                stride = stride,
                bias = False
                )
        self.bn = BatchNorm2d(out_chan, slope=slope)
        self._init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

    def _init_weight(self):
        nn.init.kaiming_normal_(self.conv.weight, a=1)
        if not self.conv.bias is None: nn.init.constant_(self.conv.bias, 0)


class ResidualBlock(nn.Module):
    def __init__(self, in_chan, out_chan, dilation=1, *args, **kwargs):
        super(ResidualBlock, self).__init__()
        inner_chan = int(out_chan / 2)
        self.conv_blk1 = ConvBNLeaky(
                in_chan,
                inner_chan,
                ks = 1,
                slope = 0.1
                )
        self.conv_blk2 = ConvBNLeaky(
                inner_chan,
                out_chan,
                ks = 3,
                pad = dilation,
                dilation = dilation,
                slope = 0.1
                )

    def forward(self, x):
        residual = self.conv_blk1(x)
        residual = self.conv_blk2(residual)
        out = x + residual
        return out


def _make_stage(in_chan, out_chan, n_block, dilation=1):
    assert dilation in (1, 2, 4)
    stride, dila_first = (2, 1) if dilation==1 else (1, dilation // 2)
    ## downsample conv
    downsample = ConvBNLeaky(
            in_chan,
            out_chan,
            ks = 3,
            stride = stride,
            pad = dila_first,
            dilation = dila_first,
            )

    layers = [downsample, ]
    for i in range(n_block):
        layers.append(ResidualBlock(
            in_chan = out_chan,
            out_chan = out_chan,
            dilation = dilation
            ))
    stage = nn.Sequential(*layers)
    return stage


class Darknet53(nn.Module):
    def __init__(self, stride=32, *args, **kwargs):
        super(Darknet53, self).__init__()
        assert stride in (8, 16, 32)
        dils = [1, 1] if stride==32 else [el * (16//stride) for el in (1, 2)]
        self.conv1 = ConvBNLeaky(3, 32, ks=3, stride=1, pad=1)
        self.layer1 = _make_stage(32, 64, 1)
        self.layer2 = _make_stage(64, 128, 2)
        self.layer3 = _make_stage(128, 256, 8)
        self.layer4 = _make_stage(256, 512, 8, dils[0])
        self.layer5 = _make_stage(512, 1024, 4, dils[1])

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.layer1(feat)
        feat4 = self.layer2(feat)
        feat8 = self.layer3(feat4)
        feat16 = self.layer4(feat8)
        feat32 = self.layer5(feat16)
        return feat4, feat8, feat16, feat32

    def get_params(self):
        bn_params = []
        non_bn_params = list(self.parameters())
        return bn_params, non_bn_params


if __name__ == '__main__':
    pass
    net = Darknet53(stride=16)
    net.cuda()

    batchsize = 2
    ten = torch.randn(batchsize, 3, 768, 768).cuda()
    criteria = nn.CrossEntropyLoss()
    lb = torch.ones((batchsize, ), dtype=torch.long).cuda()
    dense = nn.Linear(1024, 10).cuda()

    for i in range(50):
        _, _, _, out = net(ten)
        print(out.shape)
        import torch.nn.functional as F
        feat = F.avg_pool2d(out, out.size()[2:]).view((-1, 1024))
        logits = dense(feat)
        loss = criteria(logits, lb)
        loss.backward()
        print(loss)
