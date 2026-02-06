import json
import os

import cv2
import imageio
import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm

from SimpleHRNet import SimpleHRNet
from pycocotools.coco_custom import COCO
from pycocotools.cocoeval_custom import COCOeval

class SimpleDataset(data.Dataset):
    def __init__(self, db_path, cfg):
        super().__init__()
        with open(cfg.gt_path) as anno_file:
            self.anno = json.load(anno_file)
        self.file_pairs = self.wl_ll_path_pairs()
        self.is_train = False
        self.pixel_means = np.array([122.7717, 115.9465, 102.9801])  # RGB
        self.inp_res = (256, 192)
        self.bbox_extend_factor = (0.1, 0.15)
        self.img_folder = "/ExLPose/ExLPose"

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

        if self.is_train:
            # return img, targets, valid, meta
            pass
        else:
            meta = {'imgID': a['imgInfo']['imgID'], 'det_scores': a['score']}
            return image, meta

    def __len__(self):
        return len(self.anno)


# 原始索引对应名称
orig_names = [
    "right_ankle", "right_knee", "right_hip", "left_hip",
    "left_knee", "left_ankle", "pelvis", "thorax",
    "upper_neck", "head top", "right_wrist", "right_elbow",
    "right_shoulder", "left_shoulder", "left_elbow", "left_wrist"
]

orig_names = [
    "right_ankle", "right_knee", "right_hip", "left_hip",
    "left_knee", "left_ankle", "pelvis", "thorax",
    "upper_neck", "head top", "right_wrist", "right_elbow",
    "right_shoulder", "left_shoulder", "left_elbow", "left_wrist"
]
# 目标输出的名称顺序，以及它们在 orig_names 中对应的索引
new_names = [
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "head", "neck"
]
new_indices = [
    orig_names.index("left_shoulder"),
    orig_names.index("right_shoulder"),
    orig_names.index("left_elbow"),
    orig_names.index("right_elbow"),
    orig_names.index("left_wrist"),
    orig_names.index("right_wrist"),
    orig_names.index("left_hip"),
    orig_names.index("right_hip"),
    orig_names.index("left_knee"),
    orig_names.index("right_knee"),
    orig_names.index("left_ankle"),
    orig_names.index("right_ankle"),
    orig_names.index("head top"),  # 重命名为 "head"
    orig_names.index("upper_neck")  # 重命名为 "neck"
]


class CfgExtreme:
    gt_path = os.path.join('/ExLPose/Annotations/ExLPose/ExLPose_test_LL-Extreme_trans.json')
    ori_gt_path = os.path.join('/ExLPose/Annotations/ExLPose/ExLPose_test_LL-Extreme.json')
    data_shape = (256, 192)


class CfgHard:
    gt_path = os.path.join('/ExLPose/Annotations/ExLPose/ExLPose_test_LL-Hard_trans.json')
    ori_gt_path = os.path.join('/ExLPose/Annotations/ExLPose/ExLPose_test_LL-Hard.json')
    data_shape = (256, 192)


class CfgNormal:
    gt_path = os.path.join('/ExLPose/Annotations/ExLPose/ExLPose_test_LL-Normal_trans.json')
    ori_gt_path = os.path.join('/ExLPose/Annotations/ExLPose/ExLPose_test_LL-Normal.json')
    data_shape = (256, 192)


def test_4():
    # [1020, 830, 2789, 939]


    e_result = test(CfgExtreme, "extreme")
    h_result = test(CfgHard, "hard")
    n_result = test(CfgNormal, "normal")

    a = f"Extreme\t Test Acc: {e_result[0]}"
    b = f"Hard\t Test Acc: {h_result[0]}"
    c = f"Normal\t Test Acc: {n_result[0]}"

    print(a)
    print(b)
    print(c)


def test(cfg, name: str):
    test_loader = torch.utils.data.DataLoader(
        SimpleDataset(None, cfg),
        batch_size=1, shuffle=False,
        num_workers=12, pin_memory=False)

    full_result = []
    for i, (inputs, meta) in tqdm(enumerate(test_loader)):
        ids = meta['imgID'].numpy()
        det_scores = meta['det_scores']

        multi_person_joints = model.predict(inputs.squeeze(0).numpy())

        for p in range(multi_person_joints.shape[0]):
            joints = multi_person_joints[p]
            joints = joints[new_indices]

            single_result = []
            for i in range(14):
                y = float(joints[i, 0])
                x = float(joints[i, 1])
                single_result.append(x)
                single_result.append(y)
                single_result.append(1)
            if len(single_result) != 0:
                single_result_dict = {'image_id': int(ids[0]), 'category_id': 1, 'keypoints': single_result,
                                      'score': float(det_scores[0]) * joints[:, 2].mean()}
                full_result.append(single_result_dict)

    # f_name = os.path.join("uni_test", f'{save_name}-ll-{prefix}.json')
    f_name = os.path.join(f'{name}-test.json')
    with open(f_name, 'w') as wf:
        json.dump(full_result, wf)

    if len(full_result) == 0:
        return [0]
    eval_gt = COCO(cfg.ori_gt_path)
    eval_dt = eval_gt.loadRes(f_name)
    cocoEval = COCOeval(eval_gt, eval_dt, iouType='keypoints')
    cocoEval.evaluate()
    cocoEval.accumulate()
    result = cocoEval.summarize()
    return result


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleHRNet(32, 16, "./weights/pose_hrnet_w32_256x256.pth", resolution=(256, 256), yolo_version='v5',
                    yolo_model_def='yolov5n',
                    device=device, multiperson=True)
if __name__ == "__main__":
    test_4()
