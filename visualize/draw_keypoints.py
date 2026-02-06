import sys

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import cv2
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import matplotlib.pyplot as plt
import cv2
import torch

from networks import network

body = [(1, 3), (0, 2), (3, 5), (2, 4), (0, 1), (7, 9), (8, 10), (9, 11), (6, 8), (6, 7), (1, 7), (0, 6), (12, 13),
        (0, 13), (1, 13)]


def plot_keypoints_with_edges(
        img,
        keypoints,
        symmetry=body,
        figsize=(6, 8),
        point_color='lime',
        edge_color='red',
        marker='o',
        markersize=20,
        linewidth=2
):
    """
    在 matplotlib 中绘制关键点并连线。

    参数
    ----
    img         : np.ndarray, BGR 格式的图像
    keypoints   : list of float, [x1, y1, v1, x2, y2, v2, …]，已映射到 img 尺寸下
    symmetry    : list of tuple, [(i1, j1), (i2, j2), …]，关键点索引对（0-based）
    figsize     : tuple, 图像显示大小
    point_color : str, 关键点的颜色
    edge_color  : str, 连接线的颜色
    marker      : str, 点的形状
    markersize  : int, 点的大小
    linewidth   : int, 线宽
    """
    # 转为 RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=figsize)
    plt.imshow(img_rgb)

    # 提取所有可见点坐标
    xs, ys = [], []
    for idx in range(0, len(keypoints), 3):
        x, y, v = keypoints[idx:idx + 3]
        if v > 0:
            xs.append(x);
            ys.append(y)
    plt.scatter(xs, ys, c=point_color, marker=marker, s=markersize)

    # 画连线
    for (i, j) in symmetry:
        # 如果用户给的是 1-based 索引，需要减 1： i, j = i-1, j-1
        xi, yi, vi = keypoints[i * 3:i * 3 + 3]
        xj, yj, vj = keypoints[j * 3:j * 3 + 3]
        if vi > 0 and vj > 0:
            plt.plot([xi, xj], [yi, yj], c=edge_color, linewidth=linewidth)

    plt.axis('off')
    plt.tight_layout()
    plt.show()


def transform_keypoints(orig_kps, details, out_w=192, out_h=256):
    dx, dy, x2_off, y2_off = details
    crop_w = x2_off - dx
    crop_h = y2_off - dy
    x_ratio = out_w / crop_w
    y_ratio = out_h / crop_h
    kps2 = []
    for i in range(0, len(orig_kps), 3):
        x, y, v = orig_kps[i:i + 3]
        x2 = (x - dx) * x_ratio
        y2 = (y - dy) * y_ratio
        kps2 += [x2, y2, v]
    return kps2


from 生成截取图片 import norm_ds

sys.path.insert(0, "/home/wjm/MyFinalProject/pytorch-cpn/256.192.model/allconfig")
sys.path.insert(0, "/home/wjm/MyFinalProject/pytorch-cpn/utils")
from allconfig.test_config_all import cfg_test_all
from allconfig.test_config_extreme import cfg_test_extreme
from allconfig.test_config_hard import cfg_test_hard
from allconfig.test_config_normal import cfg_test_normal
from utils.imutils import *

sys.path.insert(0, "/home/wjm/MyFinalProject/pytorch-cpn/256.192.model")
from train_ll_cpn_config import cfg


def create_model(m_type):
    if m_type == "LSBN_CPN50":
        return network.__dict__[m_type](cfg.output_shape, cfg.num_class, pretrained=True)
    elif m_type == "CID_CPN50":
        return network.__dict__[m_type](cfg.output_shape, cfg.num_class,
                                        "/home/wjm/clone_projects/HVI-CIDNet/weights/LOLv1/w_perc.pth",
                                        False, pretrained=True)
    else:
        return network.__dict__[m_type](cfg.output_shape, cfg.num_class, False, pretrained=True)


model_type = "LSBN_CPN50"
# if __name__ == "__main__":
#     norm_img, meta = norm_ds[141]
#
#     norm_img = norm_img.unsqueeze(0)
#
#     meta = {k: torch.tensor(v).unsqueeze(0) for k, v in meta.items()}
#     inputs = norm_img
#     i_cfg = cfg_test_normal
#
#     model = create_model(model_type)
#     model = torch.nn.DataParallel(model).cuda()
#
#     model_path = "/home/wjm/MyFinalProject/pytorch-cpn/256.192.model/checkpoint/archive/Final_model.pth.tar"
#     checkpoint = torch.load(model_path)
#     pretrained_dict = checkpoint['state_dict']
#     model.load_state_dict(pretrained_dict)
#     cudnn.benchmark = True
#     model.eval()
#
#     with torch.no_grad():
#         input_var = torch.autograd.Variable(inputs.cuda())
#
#         flip_inputs = inputs.clone()
#         for i, finp in enumerate(flip_inputs):
#             finp = im_to_numpy(finp)
#             finp = cv2.flip(finp, 1)
#             flip_inputs[i] = im_to_torch(finp)
#         flip_input_var = torch.autograd.Variable(flip_inputs.cuda())
#         # compute output
#         if model_type == "LSBN_CPN50":
#             *_, global_outputs, refine_output = model(input_var, 0 * torch.ones(input_var.shape[0],
#                                                                                 dtype=torch.long).cuda())
#         elif model_type == "E_CPN50":
#             global_outputs, refine_output, _ = model(input_var)
#         else:
#             global_outputs, refine_output = model(input_var)
#         score_map = refine_output.data.cpu()
#         score_map = score_map.numpy()
#
#         if model_type == "LSBN_CPN50":
#             *_, flip_global_outputs, flip_output = model(flip_input_var, 0 * torch.ones(flip_input_var.shape[0],
#                                                                                         dtype=torch.long).cuda())
#         elif model_type == "E_CPN50":
#             flip_global_outputs, flip_output, _ = model(flip_input_var)
#         else:
#             flip_global_outputs, flip_output = model(flip_input_var)
#         flip_score_map = flip_output.data.cpu()
#         flip_score_map = flip_score_map.numpy()
#
#         for i, fscore in enumerate(flip_score_map):
#             fscore = fscore.transpose((1, 2, 0))
#             fscore = cv2.flip(fscore, 1)
#             fscore = list(fscore.transpose((2, 0, 1)))
#             for (q, w) in i_cfg.symmetry:
#                 fscore[q], fscore[w] = fscore[w], fscore[q]
#             fscore = np.array(fscore)
#             score_map[i] += fscore
#             score_map[i] /= 2
#
#         ids = meta['imgID'].numpy()
#         det_scores = meta['det_scores']
#
#         single_result = []
#         for b in range(inputs.size(0)):
#             details = meta['augmentation_details']
#
#             single_map = score_map[b]
#             r0 = single_map.copy()
#             r0 /= 255
#             r0 += 0.5
#             v_score = np.zeros(14)
#             for p in range(14):
#                 single_map[p] /= np.amax(single_map[p])
#                 border = 10
#                 dr = np.zeros(
#                     (i_cfg.output_shape[0] + 2 * border, i_cfg.output_shape[1] + 2 * border))
#                 dr[border:-border, border:-border] = single_map[p].copy()
#                 dr = cv2.GaussianBlur(dr, (21, 21), 0)
#                 lb = dr.argmax()
#                 y, x = np.unravel_index(lb, dr.shape)
#                 dr[y, x] = 0
#                 lb = dr.argmax()
#                 py, px = np.unravel_index(lb, dr.shape)
#                 y -= border
#                 x -= border
#                 py -= border + y
#                 px -= border + x
#                 ln = (px ** 2 + py ** 2) ** 0.5
#                 delta = 0.25
#                 if ln > 1e-3:
#                     x += delta * px / ln
#                     y += delta * py / ln
#                 x = max(0, min(x, i_cfg.output_shape[1] - 1))
#                 y = max(0, min(y, i_cfg.output_shape[0] - 1))
#                 resy = float(
#                     (4 * y + 2) / i_cfg.data_shape[0] * (details[b][3] - details[b][1]) + details[b][1])
#                 resx = float(
#                     (4 * x + 2) / i_cfg.data_shape[1] * (details[b][2] - details[b][0]) + details[b][0])
#                 v_score[p] = float(r0[p, int(round(y) + 1e-10), int(round(x) + 1e-10)])
#                 single_result.append(resx)
#                 single_result.append(resy)
#                 single_result.append(1)
#         print(single_result)
#
#         import imageio
#
#         wl = imageio.v2.imread("/home/wjm/MyFinalProject/pytorch-cpn/visualize/crop_img/normal_141_gt.png",
#                                pilmode="RGB")
#         x_k = transform_keypoints(single_result, meta['augmentation_details'].squeeze(0).numpy())
#         plot_keypoints_with_edges(wl, x_k)


_, meta = norm_ds[141]
img = cv2.imread("/home/wjm/MyFinalProject/pytorch-cpn/visualize/crop_img/normal_141_gt.png")
x_k = transform_keypoints(norm_ds.anno[141]['unit']['keypoints'], meta['augmentation_details'])
keypoints = np.array(x_k).reshape(-1, 3)
for x, y, z in keypoints:
    cv2.circle(img, (math.floor(x), math.floor(y)), 1, (0, 0, 255), 3)
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
    cv2.line(img, plt1, plt2, (255, 0, 0), 1)
cv2.imwrite("/home/wjm/MyFinalProject/pytorch-cpn/visualize/draw_line/gt.png", img)