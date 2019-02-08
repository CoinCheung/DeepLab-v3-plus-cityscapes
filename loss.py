#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = thresh
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        n_pixs = N * H * W
        logits = logits.permute(0, 2, 3, 1).contiguous().view(-1, C)
        labels = labels.view(-1)
        with torch.no_grad():
            scores = F.softmax(logits, dim=1)
            labels_cpu = labels
            invalid_mask = labels_cpu==self.ignore_lb
            labels_cpu[invalid_mask] = 0
            picks = scores[torch.arange(n_pixs), labels_cpu]
            picks[invalid_mask] = 1
            sorteds, _ = torch.sort(picks)
            thresh = self.thresh if sorteds[self.n_min]<self.thresh else sorteds[self.n_min]
            labels[picks>thresh] = self.ignore_lb
        ## TODO: here see if torch or numpy is faster
        labels = labels.clone()
        loss = self.criteria(logits, labels)
        return loss


if __name__ == '__main__':
    criteria1 = OhemCELoss(thresh=0.7, n_min=16*20*20//16).cuda()
    criteria2 = OhemCELoss(thresh=0.7, n_min=16*20*20//16).cuda()
    net1 = nn.Sequential(
        nn.Conv2d(3, 19, kernel_size=3, stride=2, padding=1),
    )
    net1.cuda()
    net1.train()
    net2 = nn.Sequential(
        nn.Conv2d(3, 19, kernel_size=3, stride=2, padding=1),
    )
    net2.cuda()
    net2.train()

    with torch.no_grad():
        inten = torch.randn(16, 3, 20, 20).cuda()
        lbs = torch.randint(0, 19, [16, 20, 20]).cuda()
        lbs[1, 10, 10] = 255

    logits1 = net1(inten)
    logits1 = F.interpolate(logits1, inten.size()[2:], mode='bilinear')
    logits2 = net2(inten)
    logits2 = F.interpolate(logits2, inten.size()[2:], mode='bilinear')

    loss1 = criteria1(logits1, lbs)
    loss2 = criteria2(logits2, lbs)
    loss = loss1 + loss2
    loss.backward()
