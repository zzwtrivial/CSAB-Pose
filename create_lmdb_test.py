import datetime
import os
import pickle
import sys

import lmdb
import numpy as np
import pyarrow as pa


class Config:
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    this_dir_name = cur_dir.split('/')[-1]
    root_dir = os.path.join(cur_dir, '..')  # cpn

    model = 'CPN50'

    lr = 5e-4
    lr_gamma = 0.5
    lr_dec_epoch = list(range(6, 40, 6))

    batch_size = 16
    weight_decay = 1e-5

    num_class = 14
    # img_path = os.path.join(root_dir, '/data01/', 'PoseInTheDark')
    img_path = "/ExLPose/ExLPose"
    symmetry = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]
    bbox_extend_factor = (0.1, 0.15)  # x, y

    # data augmentation setting
    scale_factor = (0.7, 1.35)
    rot_factor = 45

    pixel_means = np.array([122.7717, 115.9465, 102.9801])  # RGB
    data_shape = (256, 192)
    output_shape = (64, 48)
    gaussain_kernel = (7, 7)

    gk15 = (15, 15)
    gk11 = (11, 11)
    gk9 = (9, 9)
    gk7 = (7, 7)

    gt_path = os.path.join('/ExLPose/Annotations/ExLPose', 'ExLPose_train_trans.json')


import json

import torch.utils.data as data

from utils.transforms import *
import torch.utils.data as data
import imageio

sys.path.append("./256.192.model")
from allconfig.test_config_all import cfg_test_all
from allconfig.test_config_extreme import cfg_test_extreme
from allconfig.test_config_hard import cfg_test_hard
from allconfig.test_config_normal import cfg_test_normal

class TestDataSet(data.Dataset):
    def __init__(self, cfg, train=False):
        self.img_folder = cfg.img_path
        self.is_train = train
        self.inp_res = cfg.data_shape
        self.out_res = cfg.output_shape
        self.pixel_means = cfg.pixel_means
        self.num_class = cfg.num_class
        self.cfg = cfg
        self.bbox_extend_factor = cfg.bbox_extend_factor
        if train:
            self.scale_factor = cfg.scale_factor
            self.rot_factor = cfg.rot_factor
            self.symmetry = cfg.symmetry
        with open(cfg.gt_path) as anno_file:
            self.anno = json.load(anno_file)
        self.file_pairs = self.wl_ll_path_pairs()

    def wl_ll_path_pairs(self):
        pairs_file_path = "/ExLPose/Annotations/ExLPose/brightPath2darkPath.txt"
        wl_ll_pairs = {}
        with open(pairs_file_path) as f:
            x = f.readlines()
            for i in x:
                t = i.split(" ")
                # t[1].strip("\n")
                wl_ll_pairs[t[0]] = t[1].rstrip("\n")
        return wl_ll_pairs

    def augmentationCropImage(self, img, bbox, joints=None):
        height, width = self.inp_res[0], self.inp_res[1]
        bbox = np.array(bbox).reshape(4, ).astype(np.float32)
        add = max(img.shape[0], img.shape[1])
        mean_value = self.pixel_means
        bimg = cv2.copyMakeBorder(img, add, add, add, add, borderType=cv2.BORDER_CONSTANT, value=mean_value.tolist())
        objcenter = np.array([(bbox[0] + bbox[2]) / 2., (bbox[1] + bbox[3]) / 2.])
        bbox += add
        objcenter += add
        if self.is_train:
            joints[:, :2] += add
            inds = np.where(joints[:, -1] == 0)
            joints[inds, :2] = -1000000  # avoid influencing by data processing
        crop_width = (bbox[2] - bbox[0]) * (1 + self.bbox_extend_factor[0] * 2)
        crop_height = (bbox[3] - bbox[1]) * (1 + self.bbox_extend_factor[1] * 2)
        if self.is_train:
            crop_width = crop_width * (1 + 0.25)
            crop_height = crop_height * (1 + 0.25)
        if crop_height / height > crop_width / width:
            crop_size = crop_height
            min_shape = height
        else:
            crop_size = crop_width
            min_shape = width

        crop_size = min(crop_size, objcenter[0] / width * min_shape * 2. - 1.)
        crop_size = min(crop_size, (bimg.shape[1] - objcenter[0]) / width * min_shape * 2. - 1)
        crop_size = min(crop_size, objcenter[1] / height * min_shape * 2. - 1.)
        crop_size = min(crop_size, (bimg.shape[0] - objcenter[1]) / height * min_shape * 2. - 1)

        min_x = int(objcenter[0] - crop_size / 2. / min_shape * width)
        max_x = int(objcenter[0] + crop_size / 2. / min_shape * width)
        min_y = int(objcenter[1] - crop_size / 2. / min_shape * height)
        max_y = int(objcenter[1] + crop_size / 2. / min_shape * height)

        x_ratio = float(width) / (max_x - min_x)
        y_ratio = float(height) / (max_y - min_y)

        if self.is_train:
            joints[:, 0] = joints[:, 0] - min_x
            joints[:, 1] = joints[:, 1] - min_y

            joints[:, 0] *= x_ratio
            joints[:, 1] *= y_ratio
            label = joints[:, :2].copy()
            valid = joints[:, 2].copy()

        img = cv2.resize(bimg[min_y:max_y, min_x:max_x, :], (width, height))
        details = np.asarray([min_x - add, min_y - add, max_x - add, max_y - add]).astype(float)

        if self.is_train:
            return img, joints, details
        else:
            return img, details

    def __getitem__(self, index):
        a = self.anno[index]
        image_name = a['imgInfo']['img_paths']
        # LL_image
        img_path = os.path.join(self.img_folder, self.file_pairs[image_name])
        if self.is_train:
            points = np.array(a['unit']['keypoints']).reshape(self.num_class, 3).astype(np.float32)
        gt_bbox = a['unit']['GT_bbox']

        # image = scipy.misc.imread(img_path, mode='RGB')
        image = imageio.v2.imread(img_path, pilmode='RGB')
        #
        # rgb_mean_LL = np.mean(image, axis=(0, 1))
        # scaling_LL = 255 * 0.4 / rgb_mean_LL
        # image = image * scaling_LL
        image, details = self.augmentationCropImage(image, gt_bbox)
        img = im_to_torch(image)
        meta = {'imgID': a['imgInfo']['imgID'], 'augmentation_details': details, 'det_scores': a['score']}
        return img, meta

    def __len__(self):
        return len(self.anno)


def dumps_pyarrow(obj):
    """
    Serialize an object.
    Returns:
    Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()


def data2lmdb(loader, db_path='data_lmdb', name="train", write_frequency=100):
    """
    Args:
        dataloader: the general dataloader of the dataset, e.g torch.utils.data.DataLoader
        db_path: the path you want to save the lmdb file
        name: train or test
        write_frequency: Write once every ? rounds

    Returns:
        None
    """
    if not os.path.exists(db_path):
        os.makedirs(db_path)
    lmdb_path = os.path.join(db_path, "%s.lmdb" % name)
    isdir = os.path.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776 * 2, readonly=False,
                   meminit=False, map_async=True)

    txn = db.begin(write=True)
    for idx, data in enumerate(loader):
        # get data from dataloader
        pic = data

        # put data to lmdb dataset
        # {idx, (in_LDRs, in_HDRs, ref_HDRs)}
        txn.put(u'{}'.format(idx).encode('ascii'), pickle.dumps(pic))
        if idx % write_frequency == 0:
            print(f"[{idx}/{len(loader)}] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pyarrow(keys))
        txn.put(b'__len__', dumps_pyarrow(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()


if __name__ == "__main__":
    """
        训练数据存入lmdb，存入的数据为：经过augmentationCropImage方法，但没有经过data_augmentation方法
        因为data_augmentation方法有随机裁剪
    """
    # cfg = Config()
    # train_loader = torch.utils.data.DataLoader(
    #     MyData(cfg),
    #     batch_size=1, shuffle=True,
    #     num_workers=12, pin_memory=True)
    # data = MyData(cfg)
    # data2lmdb(train_loader, db_path="/ExLPose", name="train")

    """
        测试数据存入lmdb，存入数据为经过经过augmentationCropImage方法前的img，删减后的meta
    """
    type_list = [
        (cfg_test_all,      "test-ll-all-crop"),
        (cfg_test_extreme,  "test-ll-extreme-crop"),
        (cfg_test_hard,     "test-ll-hard-crop"),
        (cfg_test_normal,   "test-ll-normal-crop"),
    ]
    loader = torch.utils.data.DataLoader(
            TestDataSet(cfg_test_normal),
            batch_size=4, shuffle=False,
            num_workers=12, pin_memory=True
        )
    # for i_cfg, name in type_list:
    #     loader = torch.utils.data.DataLoader(
    #         TestDataSet(i_cfg),
    #         batch_size=1, shuffle=False,
    #         num_workers=12, pin_memory=True
    #     )
    #     data2lmdb(loader, db_path="/home/wjm/MyFinalProject", name=name)
