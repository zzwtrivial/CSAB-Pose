import pickle

import torch
import torchvision.transforms

import random

import lmdb
import pyarrow as pa
import torch.utils.data as data
import torchvision.transforms

from utils.transforms import *


class LmdbDataset(data.Dataset):
    """
    基础的Dataset，从lmdb读取数据
    """

    def __init__(self, db_path, cfg):
        self.db_path = db_path
        self.scale_factor = cfg.scale_factor
        self.img_folder = cfg.img_path
        self.inp_res = cfg.data_shape
        self.out_res = cfg.output_shape
        self.pixel_means = cfg.pixel_means
        self.num_class = cfg.num_class
        self.cfg = cfg
        self.bbox_extend_factor = cfg.bbox_extend_factor
        self.rot_factor = cfg.rot_factor
        self.symmetry = cfg.symmetry

    def open_db(self):
        self.env = lmdb.open(self.db_path, subdir=False, readonly=True)
        self.txn = self.env.begin(buffers=True)
        self.len = pa.deserialize(self.txn.get(b'__len__'))
        self.keys = pa.deserialize(self.txn.get(b'__keys__'))

    def data_augmentation(self, img_LL, label):
        height, width = img_LL.shape[0], img_LL.shape[1]
        center = (width / 2., height / 2.)
        n = label.shape[0]
        affrat = random.uniform(self.scale_factor[0], self.scale_factor[1])

        halfl_w = min(width - center[0], (width - center[0]) / 1.25 * affrat)
        halfl_h = min(height - center[1], (height - center[1]) / 1.25 * affrat)

        # 使用Tensor的裁剪操作
        img_LL = img_LL[int(center[1] - halfl_h): int(center[1] + halfl_h + 1),
                 int(center[0] - halfl_w): int(center[0] + halfl_w + 1)]
        img_LL = img_LL.permute(2, 0, 1)
        t_resize = torchvision.transforms.Resize([height, width])
        img_LL = t_resize(img_LL)
        img_LL = img_LL.float()
        # img_LL = torch.tensor(img_LL, dtype=torch.float)
        if img_LL.max() > 1:
            img_LL /= 255

        # 更新label坐标
        for i in range(n):
            label[i, 0] = (label[i, 0] - center[0]) / halfl_w * (width - center[0]) + center[0]
            label[i, 1] = (label[i, 1] - center[1]) / halfl_h * (height - center[1]) + center[1]
            # 处理label的可见性
            label[i, 2] *= ((label[i, 0] >= 0) & (label[i, 0] < width) &
                            (label[i, 1] >= 0) & (label[i, 1] < height))
        label[:, :2] //= 4
        return img_LL, label

    def __getitem__(self, idx):
        if not hasattr(self, 'txn'):
            self.open_db()

        z = self.txn.get(self.keys[idx])
        p, q = pickle.loads(z)
        p = p.squeeze(0)
        q = q.squeeze(0)
        p, q = self.data_augmentation(p, q)

        target15 = np.zeros((self.num_class, self.out_res[0], self.out_res[1]))
        target11 = np.zeros((self.num_class, self.out_res[0], self.out_res[1]))
        target9 = np.zeros((self.num_class, self.out_res[0], self.out_res[1]))
        target7 = np.zeros((self.num_class, self.out_res[0], self.out_res[1]))
        for i in range(self.num_class):
            if q[i, 2] > 0:  # COCO visible: 0-no label, 1-label + invisible, 2-label + visible
                target15[i] = generate_heatmap(target15[i], q[i], self.cfg.gk15)
                target11[i] = generate_heatmap(target11[i], q[i], self.cfg.gk11)
                target9[i] = generate_heatmap(target9[i], q[i], self.cfg.gk9)
                target7[i] = generate_heatmap(target7[i], q[i], self.cfg.gk7)

        targets = [torch.Tensor(target15), torch.Tensor(target11), torch.Tensor(target9), torch.Tensor(target7)]
        valid = q[:, 2]
        return p, targets, valid

    def __len__(self):
        if not hasattr(self, 'txn'):
            self.open_db()
        return self.len
