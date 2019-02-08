#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import *
from models.deeplabv3plus import Deeplab_v3plus
from cityscapes import CityScapes

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist

import os
import sys
import os.path as osp
import logging
import time
import numpy as np
from tqdm import tqdm
import numba



@numba.jit
def compute_hist(pred, lb):
    n_classes = 19
    ignore_idx = 255
    keep = np.logical_not(lb==255)
    merge = pred[keep] * n_classes + lb[keep]
    hist = np.bincount(merge, minlength=n_classes**2)
    hist = hist.reshape((n_classes, n_classes))
    return hist


class MscEval(object):
    def __init__(self,
            ds_pth = './data',
            scales = [0.5, 0.75, 1, 1.25, 1.5, 1.75],
            n_classes = 19,
            lb_ignore = 255,
            flip = True,
            batchsize = 2,
            n_workers = 2,
            *args, **kwargs):
        self.scales = scales
        self.n_classes = n_classes
        self.lb_ignore = lb_ignore
        self.flip = flip
        ## dataloader
        dsval = CityScapes(ds_pth, mode='val')
        self.dl = DataLoader(dsval,
                        batch_size = batchsize,
                        shuffle = False,
                        num_workers = n_workers,
                        drop_last = False)


    def __call__(self, net):
        ## evaluate
        hist = np.zeros((19, 19), dtype=np.float32)
        diter = enumerate(tqdm(self.dl))
        if dist.is_initialized() and dist.get_rank()!=0:
            diter = enumerate(self.dl)
        for i, (imgs, label) in diter:
            N, _, H, W = label.shape
            probs = torch.zeros((N, self.n_classes, H, W))
            probs.requires_grad = False
            for sc in self.scales:
                new_hw = [int(H*sc), int(W*sc)]
                with torch.no_grad():
                    im = F.interpolate(imgs, new_hw, mode='bilinear', align_corners=True)
                    im = im.cuda()
                    out = net(im)
                    out = F.interpolate(out, (H, W), mode='bilinear', align_corners=True)
                    prob = F.softmax(out, 1)
                    probs += prob.cpu()
                    if self.flip:
                        out = net(torch.flip(im, dims=(3,)))
                        out = torch.flip(out, dims=(3,))
                        out = F.interpolate(out, (H, W), mode='bilinear',
                                align_corners=True)
                        prob = F.softmax(out, 1)
                        probs += prob.cpu()
                    del out, prob
            probs = probs.data.numpy()
            preds = np.argmax(probs, axis=1)

            hist_once = compute_hist(preds, label.data.numpy().squeeze(1))
            hist = hist + hist_once
        IOUs = np.diag(hist) / (np.sum(hist, axis=0)+np.sum(hist, axis=1)-np.diag(hist))
        mIOU = np.mean(IOUs)
        return mIOU


def evaluate():
    respth = './res'
    FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
    log_level = logging.INFO
    if dist.is_initialized() and dist.get_rank()!=0:
        log_level = logging.ERROR
    logging.basicConfig(level=log_level, format=FORMAT, stream=sys.stdout)
    logger = logging.getLogger()

    ## model
    logger.info('setup and restore model')
    n_classes = 19
    net = Deeplab_v3plus(n_classes=n_classes)
    save_pth = osp.join(respth, 'model_final.pth')
    net.load_state_dict(torch.load(save_pth))
    net.cuda()
    net.eval()

    ## evaluator
    logger.info('compute the mIOU')
    evaluator = MscEval()
    mIOU = evaluator(net)
    logger.info('mIOU is: {:.6f}'.format(mIOU))


if __name__ == "__main__":
    evaluate()
