import json
import math

import cv2
import os
import numpy as np
from tqdm import tqdm

result_path = "OCN_render/6"
dataset_path = "/ExLPose/ExLPose-OC"

with open(
        "/home/wjm/clone_projects/ExLPose/pytorch-cpn/256.192.model/Annotations/ExLPose-OCN/ExLPose-OC_test_RICOH3_trans.json") as gt:
    ll_normal_trans = json.load(gt)
    ll_normal_trans_dict = {x['imgInfo']['imgID']: x['imgInfo']['img_paths'].replace("PID_", "") for x in
                            ll_normal_trans}
    # xxx = [{'image_id': x['imgInfo']['imgID'], 'keypoints': x['unit']['keypoints']} for x in ll_normal_trans if x['imgInfo']['imgID'] in [66, 115, 148]]

with open("/home/wjm/MyFinalProject/pytorch-cpn/256.192.model/uni_test/OC/other6-ll-.json") as f:
    result_ll_normal = json.load(f)

for test1 in tqdm(result_ll_normal):

    file_path = os.path.join(dataset_path, ll_normal_trans_dict[test1['image_id']])
    target_path = os.path.join(result_path, str(test1['image_id']) + ".jpg")

    if os.path.exists(target_path):
        img = cv2.imread(target_path)
    else:
        img = cv2.imread(file_path)

    keypoints = np.array(test1['keypoints'])
    keypoints = keypoints.reshape(-1, 3)

    for x, y, z in keypoints:
        cv2.circle(img, (math.floor(x), math.floor(y)), 1, (0, 0, 255), 5)

    line_dict = [
        [12, 13],
        [13, 0], [0, 2], [2, 4],
        [13, 1], [1, 3], [3, 5],
        [13, 7], [7, 9], [9, 11],
        [13, 6], [6, 8], [8, 10]
    ]

    for line in line_dict:
        plt1 = (math.floor(keypoints[line[0]][0]), math.floor(keypoints[line[0]][1]))
        plt2 = (math.floor(keypoints[line[1]][0]), math.floor(keypoints[line[1]][1]))
        if plt1 ==(0,0) or plt2 == (0,0):
            continue
        cv2.line(img, plt1, plt2, (255, 255, 255), 5)

    cv2.imwrite(os.path.join(result_path, str(test1['image_id']) + ".jpg"), img)
    # print(test1['image_id'])
