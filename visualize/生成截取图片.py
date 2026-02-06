import math
import os
import sys

import cv2
import imageio
import numpy as np
from matplotlib import pyplot as plt
from torch.utils import data
from tqdm import tqdm

from utils.imutils import im_to_torch

sys.path.insert(0, "../256.192.model")
from allconfig.test_config_normal import cfg_test_normal
from allconfig.test_config_hard import cfg_test_hard
from allconfig.test_config_extreme import cfg_test_extreme
import json

from networks.other.lime.exposure_enhancement import enhance_image_exposure


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

        image = imageio.v2.imread(img_path, pilmode='RGB')

        image, details = self.augmentationCropImage(image, gt_bbox)
        img = im_to_torch(image)
        meta = {'imgID': a['imgInfo']['imgID'], 'augmentation_details': details, 'det_scores': a['score']}
        return img, meta

    def __len__(self):
        return len(self.anno)


ext_ds = TestDataSet(cfg_test_normal)
hard_ds = TestDataSet(cfg_test_hard)
norm_ds = TestDataSet(cfg_test_normal)

# if __name__ == "__main__":
#     norm_img, gt_box = norm_ds[141]
#
#     # lime_img = enhance_image_exposure(norm_img, 0.6, 0.15, False, 3,
#     #                                   1, 1, 1, 1e-3)
#
#     cv2.imwrite(f"./crop_img/normal_141_cropped.png", cv2.cvtColor(norm_img, cv2.COLOR_RGB2BGR))
#     plt.imshow(norm_img)
#     plt.show()

    # for i, (img, gtb) in tqdm(enumerate(norm_ds)):
    #     size = math.fabs((gtb[0] - gtb[2]) * (gtb[1] - gtb[3]))
    #     # print(size)
    #     if size > 300000:
    #         cv2.imwrite(f"./crop_img/{i}_cropped.png", img)
