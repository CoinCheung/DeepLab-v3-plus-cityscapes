#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import *
from models.deeplabv3plus import Deeplab_v3plus
from cityscapes import CityScapes
from configs import config_factory

import torch
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



class MscEval(object):
    def __init__(self, cfg, *args, **kwargs):
        self.cfg = cfg
        ## dataloader
        dsval = CityScapes(cfg, mode='val')
        self.dl = DataLoader(dsval,
                        batch_size = cfg.eval_batchsize,
                        shuffle = False,
                        num_workers = cfg.eval_n_workers,
                        drop_last = False)


    def __call__(self, net):
        ## evaluate
        hist_size = (self.cfg.n_classes, self.cfg.n_classes)
        hist = np.zeros(hist_size, dtype=np.float32)
        diter = enumerate(tqdm(self.dl))
        if dist.is_initialized() and dist.get_rank()!=0:
            diter = enumerate(self.dl)
        for i, (imgs, label) in diter:
            N, _, H, W = label.shape
            probs = torch.zeros((N, self.cfg.n_classes, H, W))
            probs.requires_grad = False
            for sc in self.cfg.eval_scales:
                new_hw = [int(H*sc), int(W*sc)]
                with torch.no_grad():
                    im = F.interpolate(imgs, new_hw, mode='bilinear', align_corners=True)
                    im = im.cuda()
                    out = net(im)
                    out = F.interpolate(out, (H, W), mode='bilinear', align_corners=True)
                    prob = F.softmax(out, 1)
                    probs += prob.cpu()
                    if self.cfg.eval_flip:
                        out = net(torch.flip(im, dims=(3,)))
                        out = torch.flip(out, dims=(3,))
                        out = F.interpolate(out, (H, W), mode='bilinear',
                                align_corners=True)
                        prob = F.softmax(out, 1)
                        probs += prob.cpu()
                    del out, prob
            probs = probs.data.numpy()
            preds = np.argmax(probs, axis=1)

            hist_once = self.compute_hist(preds, label.data.numpy().squeeze(1))
            hist = hist + hist_once
        IOUs = np.diag(hist) / (np.sum(hist, axis=0)+np.sum(hist, axis=1)-np.diag(hist))
        mIOU = np.mean(IOUs)
        return mIOU

    @numba.jit
    def compute_hist(self, pred, lb):
        n_classes = self.cfg.n_classes
        keep = np.logical_not(lb==self.cfg.ignore_label)
        merge = pred[keep] * n_classes + lb[keep]
        hist = np.bincount(merge, minlength=n_classes**2)
        hist = hist.reshape((n_classes, n_classes))
        return hist


def evaluate():
    ## logger
    FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
    log_level = logging.INFO
    if dist.is_initialized() and dist.get_rank()!=0:
        log_level = logging.ERROR
    logging.basicConfig(level=log_level, format=FORMAT, stream=sys.stdout)
    logger = logging.getLogger()
    cfg = config_factory['resnet_cityscapes']

    ## model
    logger.info('setup and restore model')
    net = Deeplab_v3plus(n_classes=cfg.n_classes)
    save_pth = osp.join(cfg.respth, 'model_final.pth')
    net.load_state_dict(torch.load(save_pth))
    net.cuda()
    net.eval()

    ## evaluator
    logger.info('compute the mIOU')
    evaluator = MscEval(cfg)
    mIOU = evaluator(net)
    logger.info('mIOU is: {:.6f}'.format(mIOU))


if __name__ == "__main__":
    evaluate()
