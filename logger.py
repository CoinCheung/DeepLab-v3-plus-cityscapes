#!/usr/bin/python
# -*- encoding: utf-8 -*-


import os.path as osp
import time
import sys
import logging

import torch.distributed as dist


def setup_logger(logpth):
    logfile = 'Deeplab_v3plus-{}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
    logfile = osp.join(logpth, logfile)
    FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
    if dist.get_rank()==0:
        logging.basicConfig(level=logging.INFO, format=FORMAT, filename=logfile)
    else:
        logging.basicConfig(level=logging.ERROR, format=FORMAT, filename=logfile)
    logging.root.addHandler(logging.StreamHandler())


