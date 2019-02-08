#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os.path as osp
import os
from PIL import Image
import numpy as np
import json

from transform import *



class CityScapes(Dataset):
    def __init__(self, rootpth, cropsize=(320, 240), mode='train', *args, **kwargs):
        super(CityScapes, self).__init__(*args, **kwargs)
        assert mode in ('train', 'val', 'test')
        self.mode = mode
        self.n_classes = 19

        with open('./cityscapes_info.json', 'r') as fr:
            labels_info = json.load(fr)
        self.lb_map = {el['id']: el['trainId'] for el in labels_info}

        ## parse img directory
        self.imgs = {}
        imgnames = []
        impth = osp.join(rootpth, 'leftImg8bit', mode)
        folders = os.listdir(impth)
        for fd in folders:
            fdpth = osp.join(impth, fd)
            im_names = os.listdir(fdpth)
            names = [el.replace('_leftImg8bit.png', '') for el in im_names]
            impths = [osp.join(fdpth, el) for el in im_names]
            imgnames.extend(names)
            self.imgs.update(dict(zip(names, impths)))

        ## parse gt directory
        self.labels = {}
        gtnames = []
        gtpth = osp.join(rootpth, 'gtFine', mode)
        folders = os.listdir(gtpth)
        for fd in folders:
            fdpth = osp.join(gtpth, fd)
            lbnames = os.listdir(fdpth)
            lbnames = [el for el in lbnames if 'labelIds' in el]
            names = [el.replace('_gtFine_labelIds.png', '') for el in lbnames]
            lbpths = [osp.join(fdpth, el) for el in lbnames]
            gtnames.extend(names)
            self.labels.update(dict(zip(names, lbpths)))

        self.imnames = imgnames
        self.len = len(self.imnames)
        assert set(imgnames) == set(gtnames)
        assert set(self.imnames) == set(self.imgs.keys())
        assert set(self.imnames) == set(self.labels.keys())

        ## pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        self.trans = Compose([
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            HorizontalFlip(),
            RandomScale((0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
            RandomCrop(cropsize)
            ])


    def __getitem__(self, idx):
        fn  = self.imnames[idx]
        impth = self.imgs[fn]
        lbpth = self.labels[fn]
        img = Image.open(impth)
        label = Image.open(lbpth)
        if self.mode == 'train':
            im_lb = dict(im = img, lb = label)
            im_lb = self.trans(im_lb)
            img, label = im_lb['im'], im_lb['lb']
            imgs = self.to_tensor(img)
        else:
            imgs = self.to_tensor(img)
        label = np.array(label).astype(np.int64)[np.newaxis, :]
        label = self.convert_labels(label)
        return imgs, label


    def __len__(self):
        return self.len


    def convert_labels(self, label):
        for k, v in self.lb_map.items():
            label[label == k] = v
        return label


def dl_collate_fn(batch):
    imgs, label = [], []
    for im, lb in batch:
        imgs.append(im)
        label.append(lb)
    label = np.array(label)
    ims_scale = []
    n_scales = len(imgs[0])
    for i in range(n_scales):
        scales_tens = torch.stack([el[i] for el in imgs])
        ims_scale.append(scales_tens)
    return ims_scale, label


if __name__ == "__main__":
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    ds = CityScapes('./data/', mode='val')
    #  ds = CityScapes('./data/', n_classes=19, mode='train')
    #  im, label = ds[10]
    #  _ = input()
    dl = DataLoader(ds,
                    batch_size = 4,
                    shuffle = True,
                    num_workers = 4,
                    collate_fn = dl_collate_fn,
                    drop_last = True)
    for imgs, label in dl:
        print(len(imgs))
        for el in imgs:
            print(el.size())
        #  print(imgs_flip[0].size())
        break
    #  uni = []
    #  for im, lb in tqdm(ds):
    #      lb_uni = np.unique(lb).tolist()
    #      uni.extend(lb_uni)
    #      print(len(im))
    #      for el in im:
    #          print(el.size())
    #      break
    #  print(uni)
    #  print(set(uni))
