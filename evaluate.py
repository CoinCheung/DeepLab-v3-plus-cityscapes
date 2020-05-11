#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import *
from models.deeplabv3plus import Deeplab_v3plus
from cityscapes import CityScapes
from configs import config_factory
from one_hot import convert_to_one_hot

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
import argparse



def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument(
            '--local_rank',
            dest = 'local_rank',
            type = int,
            default = -1,
            )
    return parse.parse_args()


class MscEval(object):
    def __init__(self, cfg, *args, **kwargs):
        self.cfg = cfg
        self.distributed = dist.is_initialized()
        ## dataloader
        dsval = CityScapes(cfg, mode='val')
        sampler = None
        if self.distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dsval)
        self.dl = DataLoader(dsval,
                        batch_size = cfg.eval_batchsize,
                        sampler = sampler,
                        shuffle = False,
                        num_workers = cfg.eval_n_workers,
                        drop_last = False)


    def __call__(self, net):
        ## evaluate
        n_classes = self.cfg.n_classes
        ignore_label = self.cfg.ignore_label
        if dist.is_initialized() and dist.get_rank()!=0:
            diter = enumerate(self.dl)
        else:
            diter = enumerate(tqdm(self.dl))
        hist = torch.zeros(n_classes, n_classes).cuda()
        for i, (imgs, label) in diter:
            label = label.squeeze(1).cuda()
            N, H, W = label.shape
            probs = torch.zeros((N, n_classes, H, W)).cuda()
            probs.requires_grad = False
            for sc in self.cfg.eval_scales:
                new_hw = [int(H*sc), int(W*sc)]
                with torch.no_grad():
                    im = F.interpolate(imgs, new_hw, mode='bilinear', align_corners=True)
                    im = im.cuda()
                    out = net(im)
                    out = F.interpolate(out, (H, W), mode='bilinear', align_corners=True)
                    prob = F.softmax(out, 1)
                    probs += prob
                    if self.cfg.eval_flip:
                        out = net(torch.flip(im, dims=(3,)))
                        out = torch.flip(out, dims=(3,))
                        out = F.interpolate(out, (H, W), mode='bilinear',
                                align_corners=True)
                        prob = F.softmax(out, 1)
                        probs += prob
                    del out, prob
            torch.cuda.empty_cache()
            preds = torch.argmax(probs, dim=1)
            keep = label != ignore_label
            hist += torch.bincount(
                label[keep] * n_classes + preds[keep],
                minlength=n_classes ** 2
                ).view(n_classes, n_classes)
        if dist.is_initialized():
            dist.all_reduce(hist, dist.ReduceOp.SUM)
        ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
        miou = ious.mean()
        return miou.item()


def evaluate():
    ## setup
    cfg = config_factory['resnet_cityscapes']
    args = parse_args()
    if not args.local_rank == -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(
                    backend = 'nccl',
                    init_method = 'tcp://127.0.0.1:{}'.format(cfg.port),
                    world_size = torch.cuda.device_count(),
                    rank = args.local_rank
                    )
        setup_logger(cfg.respth)
    else:
        FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
        log_level = logging.INFO
        if dist.is_initialized() and dist.get_rank()!=0:
            log_level = logging.ERROR
        logging.basicConfig(level=log_level, format=FORMAT, stream=sys.stdout)
    logger = logging.getLogger()

    ## model
    logger.info('setup and restore model')
    net = Deeplab_v3plus(cfg)
    save_pth = osp.join(cfg.respth, 'model_final.pth')
    net.load_state_dict(torch.load(save_pth), strict=False)
    net.cuda()
    net.eval()
    if not args.local_rank == -1:
        net = nn.parallel.DistributedDataParallel(net,
                device_ids = [args.local_rank, ],
                output_device = args.local_rank
                )

    ## evaluator
    logger.info('compute the mIOU')
    evaluator = MscEval(cfg)
    mIOU = evaluator(net)
    logger.info('mIOU is: {:.6f}'.format(mIOU))


if __name__ == "__main__":
    evaluate()
