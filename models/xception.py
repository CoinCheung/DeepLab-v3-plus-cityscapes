#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.utils.model_zoo as modelzoo
import torch.nn.functional as F
import torchvision
import torch.utils.checkpoint as ckpt

from modules import InPlaceABNSync as BatchNorm2d


class ConvBNReLU(nn.Module):
    def __init__(self,
            in_chan,
            out_chan,
            ks = 3,
            stride = 1,
            padding = 1,
            bias = False,
            *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = bias)
        self.bn = BatchNorm2d(out_chan)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class SepConvBNReLU(nn.Module):
    def __init__(self,
            in_chan,
            out_chan,
            ks = 3,
            stride = 1,
            padding = 1,
            dilation = 1,
            bias = False,
            *args, **kwargs):
        super(SepConvBNReLU, self).__init__()

        self.conv = nn.Conv2d(in_chan,
                in_chan,
                kernel_size = ks,
                stride = stride,
                padding = dilation,
                dilation = dilation,
                groups = in_chan,
                bias = bias)
        self.bn = BatchNorm2d(in_chan)
        self.pairwise = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = 1,
                bias = bias)

        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.pairwise(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class Block(nn.Module):
    def __init__(self,
            in_chan,
            out_chan,
            reps = 3,
            stride = 1,
            dilation = 1,
            start_with_relu = True,
            grow_layer = 0,
            bias = False,
            *args, **kwargs):
        super(Block, self).__init__()

        self.stride = stride
        self.in_chan = in_chan
        self.out_chan = out_chan
        # compute channels of each layer
        inchans = [in_chan] * reps
        outchans = [out_chan] * reps
        for i in range(reps):
            if i < grow_layer:
                inchans[i] = in_chan
                outchans[i] = in_chan
            else:
                inchans[i] = out_chan
                outchans[i] = out_chan
        inchans[grow_layer] = in_chan
        outchans[grow_layer] = out_chan

        # shortcut
        self.shortcut = nn.Sequential(*[
                nn.Conv2d(in_chan,
                    out_chan,
                    kernel_size = 1,
                    stride = stride,
                    bias = bias),
                nn.BatchNorm2d(out_chan),
            ])

        # residual
        layers = []
        if start_with_relu: layers.append(nn.ReLU(inplace=False))
        for i in range(reps-1):
            layers.append(SepConvBNReLU(inchans[i],
                    outchans[i],
                    kernel_size = 3,
                    stride = 1,
                    padding = dilation,
                    dilation = dilation,
                    bias = bias))
            layers.append(nn.BatchNorm2d(outchans[i]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(SepConvBNReLU(inchans[-1],
                outchans[-1],
                kernel_size = 3,
                stride = stride, # stride always at the last
                padding = dilation,
                dilation = dilation,
                bias = bias))
        layers.append(nn.BatchNorm2d(outchans[i]))
        self.residual = nn.Sequential(*layers)

        self.init_weight()

    def forward(self, x):
        resd = self.residual(x)
        if not self.stride == 1 or not self.in_chan == self.out_chan:
            sc = self.shortcut(x)
        else:
            sc = x
        out = resd + sc
        return out

    def init_weight(self):
        for ly in self.shortcut.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class EntryFlow(nn.Module):
    def __init__(self, *args, **kwargs):
        super(EntryFlow, self).__init__()
        self.conv1 = ConvBNReLU(3, 32, ks=3, stride=2, padding=1) # 1/2
        self.conv2 = ConvBNReLU(32, 64, ks=3, stride=1, padding=1)
        self.block1 = Block(
            64,
            128,
            reps = 3,
            stride = 2,
            dilation = 1,
            start_with_relu = False,
            grow_layer = 0,) # 1/4
        self.block2 = Block(
            128,
            256,
            reps = 3,
            stride = 1,
            dilation = 1,
            start_with_relu = True,
            grow_layer = 0,)
        self.block3 = Block(
            256,
            256,
            reps = 3,
            stride = 2,
            dilation = 1,
            start_with_relu = True,
            grow_layer = 0,) # 1/8
        self.block4 = Block(
            256,
            728,
            reps = 3,
            stride = 1,
            dilation = 1,
            start_with_relu = True,
            grow_layer = 0,)
        self.block5 = Block(
            728,
            728,
            reps = 3,
            stride = 2,
            dilation = 1,
            start_with_relu = True,
            grow_layer = 0,)

    def forward(self, x):
        feat2 = self.conv1(x)
        feat2 = self.conv2(feat2)
        feat4 = self.block1(feat2)
        feat4 = self.block2(feat4)
        feat8 = self.block3(feat4)
        feat8 = self.block4(feat8)
        feat16 = self.block5(feat8)
        return feat4, feat16


class MiddleFlow(nn.Module):
    def __init__(self, dilation=1, *args, **kwargs):
        super(MiddleFlow, self).__init__()
        middle_layers = []
        for i in range(16):
            middle_layers.append(Block(
                728,
                728,
                reps = 3,
                stride = 1,
                dilation = dilation,
                start_with_relu = True,
                grow_layer = 0,))
        self.middle_flow = nn.Sequential(*middle_layers)

    def forward(self, x):
        out = self.middle_flow(x)
        return out


class ExitEntry(nn.Module):
    def __init__(self, stride=1, dilation=2, *args, **kwargs):
        super(ExitEntry, self).__init__()
        self.block1 = Block(
            728,
            1024,
            reps = 3,
            stride = stride,
            dilation = dilation,
            start_with_relu = True,
            grow_layer = 1,)
        self.sepconv1 = SepConvBNReLU(
            1024,
            1536,
            kernel_size = 3,
            stride = 1,
            padding = dilation,
            dilation = dilation,
            bias = False)
        self.sepconv2 = SepConvBNReLU(
            1536,
            1536,
            kernel_size = 3,
            stride = 1,
            padding = dilation,
            dilation = dilation,
            bias = False)
        self.sepconv3 = SepConvBNReLU(
            1536,
            2048,
            kernel_size = 3,
            stride = 1,
            padding = dilation,
            dilation = dilation,
            bias = False)

    def forward(self, x):
        x = self.block1(x)
        x = self.sepconv1(x)
        x = self.sepconv2(x)
        out = self.sepconv3(x)
        return out



class Xception71(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Xception71, self).__init__()

        self.entry_flow = EntryFlow()
        self.middle_flow = MiddleFlow(dilation=1)
        self.exit_flow = ExitEntry(stride=1, dilation=2)

    def forward(self, x):
        feat4, feat16 = self.entry_flow(x)
        #  feat_mid = ckpt.checkpoint(self.middle_flow, feat16)
        feat_mid = self.middle_flow(feat16)
        feat_exit = self.exit_flow(feat_mid)
        return feat4, feat_exit



if __name__ == "__main__":
    net = Xception71()
    net.train()
    net.cuda()
    Loss = nn.CrossEntropyLoss(ignore_index=255)
    import numpy as np
    inten = torch.tensor(np.random.randn(16, 3, 320, 240).astype(np.float32), requires_grad=False).cuda()
    label = torch.randint(0, 10, (16,)).cuda()
    for i in range(100):
        feat4, out = net(inten)
        logits = F.avg_pool2d(out, out.size()[2:]).view((16, -1))
        scores = F.softmax(logits, 1)
        loss = Loss(scores, label)
        loss.backward()
        print(i)
        print(out.size())

