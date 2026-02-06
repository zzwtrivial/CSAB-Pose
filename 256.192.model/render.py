import argparse
import json
import math

import cv2
import os
import numpy as np
from tqdm import tqdm


def convert_coco17_to_custom14(keypoints_array: np.ndarray) -> np.ndarray:
    """
    Convert a COCO 17 keypoints format (17, 3) to a custom 14 keypoints format (14, 3).
    The custom format includes:
        - 12 reordered keypoints
        - "head" as average of nose, eyes, and ears
        - "neck" as midpoint of shoulders

    Args:
        keypoints_array (np.ndarray): Input array of shape (17, 3)

    Returns:
        np.ndarray: Output array of shape (14, 3)
    """
    orig_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
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
        orig_names.index("right_ankle")
    ]

    selected = keypoints_array[new_indices]

    # Calculate head: average of visible [nose, left_eye, right_eye, left_ear, right_ear]
    head_ids = [orig_names.index(k) for k in ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]]
    head_pts = keypoints_array[head_ids]
    mask = head_pts[:, 2] > 0
    if mask.any():
        head_x = head_pts[mask, 0].mean()
        head_y = head_pts[mask, 1].mean()
        head_v = 2 if (head_pts[mask, 2] == 2).any() else 1
    else:
        head_x, head_y, head_v = 0, 0, 0
    head = np.array([head_x, head_y, head_v])

    # Calculate neck: midpoint of visible [left_shoulder, right_shoulder]
    neck_ids = [orig_names.index("left_shoulder"), orig_names.index("right_shoulder")]
    neck_pts = keypoints_array[neck_ids]
    mask = neck_pts[:, 2] > 0
    if mask.any():
        neck_x = neck_pts[mask, 0].mean()
        neck_y = neck_pts[mask, 1].mean()
        neck_v = 2 if (neck_pts[mask, 2] == 2).any() else 1
    else:
        neck_x, neck_y, neck_v = 0, 0, 0
    neck = np.array([neck_x, neck_y, neck_v])

    return np.vstack([selected, head, neck])


parser = argparse.ArgumentParser()
parser.add_argument('--result', required=True, type=str)
parser.add_argument('--anno', type=str, required=True,
                    help='Experiment name')
args = parser.parse_args()

result_path = "result_LL_normal/chapter3"
dataset_path = "/ExLPose/ExLPose"

result_path = args.result
anno_path = args.anno

with open("/ExLPose/Annotations/ExLPose/ExLPose_test_LL-Normal_trans.json") as gt:
    ll_normal_trans = json.load(gt)
    ll_normal_trans_dict = {x['imgInfo']['imgID']: x['imgInfo']['img_paths'] for x in ll_normal_trans}

with open(anno_path) as f:
    result_ll_normal = json.load(f)


def wl_ll_path_pairs():
    pairs_file_path = "/ExLPose/Annotations/ExLPose/brightPath2darkPath.txt"
    wl_ll_pairs = {}
    with open(pairs_file_path) as f:
        x = f.readlines()
        for i in x:
            t = i.split(" ")
            # t[1].strip("\n")
            wl_ll_pairs[t[0]] = t[1].rstrip("\n")
    return wl_ll_pairs


pairs = wl_ll_path_pairs()

for test1 in tqdm(result_ll_normal):

    file_path = os.path.join(dataset_path, pairs[ll_normal_trans_dict[test1['image_id']]])
    target_path = os.path.join(result_path, str(test1['image_id']) + ".jpg")

    if os.path.exists(target_path):
        img = cv2.imread(target_path)
    else:
        img = cv2.imread(file_path)

    keypoints = np.array(test1['keypoints'])
    if len(keypoints) == 17 * 3:
        keypoints = convert_coco17_to_custom14(keypoints)
    keypoints = keypoints.reshape(-1, 3)

    for idx, (x, y, z) in enumerate(keypoints):
        x_int = int(math.floor(x))
        y_int = int(math.floor(y))

        # 画点
        cv2.circle(img, (x_int, y_int), 3, (0, 0, 255), -1)

        # 标号（编号）
        cv2.putText(
            img,
            str(idx),
            (x_int + 5, y_int - 5),  # 偏移一点防止挡住圆点
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
            cv2.LINE_AA
        )

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
        if plt1 == (0, 0) or plt2 == (0, 0):
            continue
        cv2.line(img, plt1, plt2, (255, 0, 0), 3)

    cv2.imwrite(os.path.join(result_path, str(test1['image_id']) + ".jpg"), img)
    # print(test1['image_id'])
