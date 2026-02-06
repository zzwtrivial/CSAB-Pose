import argparse
import sys
from math import ceil

import cv2
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
from matplotlib import pyplot as plt
from torchvision.utils import make_grid

sys.path.insert(0, "../256.192.model")

from uni_test import create_model
import pytorch_grad_cam
from pytorch_grad_cam.utils.image import show_cam_on_image

import numpy as np
import matplotlib.pyplot as plt


def visualize_feature_maps(feature_map, save_path=None):
    """
    可视化 shape 为 [1, 64, H, W] 的特征图，按通道分别展示为 8x8 子图。

    参数：
        feature_map (np.ndarray): 输入的特征图，shape 应为 [1, 64, H, W]。
        save_path (str, optional): 如果提供路径，则保存图片到该路径；否则只展示。
    """
    if feature_map.shape[0] != 1 or feature_map.shape[1] != 64:
        raise ValueError("特征图必须是 shape=[1, 64, H, W] 的数组。")

    feature_map = feature_map[0]  # 去除 batch 维度，shape 变为 [64, H, W]
    H, W = feature_map.shape[1], feature_map.shape[2]

    fig, axs = plt.subplots(8, 8, figsize=(16, 16))
    fig.suptitle('Feature Maps by Channel', fontsize=16)

    for i in range(64):
        ax = axs[i // 8, i % 8]
        ax.imshow(feature_map[i], cmap='viridis')
        ax.axis('off')
        ax.set_title(f'Ch {i}', fontsize=8)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


def fuse_max(feature_map):
    """
    feature_map: torch.Tensor, shape [B, C, H, W]
    返回: torch.Tensor, shape [B, H, W]
    """
    # 在通道维度做最大值
    return feature_map.max(dim=1)[0]


def fuse_mean(feature_map):
    """
    feature_map: torch.Tensor, shape [B, C, H, W]
    返回: torch.Tensor, shape [B, H, W]
    """
    # 在通道维度做平均
    return feature_map.mean(dim=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str,
                        help='Model name or path')
    parser.add_argument('--path', type=str, required=True,
                        help='Experiment name')
    parser.add_argument('--save', type=str, required=True,
                        help='Experiment name')
    parser.add_argument('--img', type=str, required=False, default="crop_img/normal_141_cropped_ps.png",
                        help='Experiment name')
    args = parser.parse_args()

    model_type = args.model
    model = create_model(model_type)
    model = torch.nn.DataParallel(model).cuda()

    model_path = args.path
    checkpoint = torch.load(model_path)
    pretrained_dict = checkpoint['state_dict']
    model.load_state_dict(pretrained_dict)
    cudnn.benchmark = True
    model.eval()

    import imageio

    img = imageio.v2.imread(args.img, pilmode="RGB")
    img = torch.from_numpy(img)
    img = img.permute([2, 0, 1]).unsqueeze(0) / 255

    # 1：定义module_name用于记录相应的module名字、定义用于获取网络各层输入输出tensor的容器
    module_name = []
    features_in_hook = []
    features_out_hook = []


    # 2：hook函数负责将相应的module名字、获取的输入输出 添加到feature列表中
    def hook(module, fea_in, fea_out):
        print("hooker working")
        module_name.append(module.__class__)
        features_in_hook.append(fea_in)
        features_out_hook.append(fea_out)
        return None


    # x = model.module.cpn.resnet.layer1.register_forward_hook(hook)
    x = model.module.resnet_lsbn.layer1.register_forward_hook(hook)
    # x = model.module.resnet.layer1.register_forward_hook(hook)
    model(img, torch.zeros(img.shape[0], dtype=torch.long))

    x = features_in_hook[0][0].detach().cpu()

    # visualize_feature_maps(x)

    fused = fuse_mean(x)[0].cpu().numpy()

    dpi = 100
    scale = 5
    h, w = fused.shape
    figsize = (scale * w / dpi, scale * h / dpi)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(fused, cmap='jet')
    plt.savefig(args.save, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close()

    # x = x.permute([1,0,2,3])
    # a = make_grid(x, padding=0, normalize=True, scale_each=True, pad_value=1)
    # gray_grid = a[0]
    # plt.figure(figsize=(8, 8))
    # plt.imshow(gray_grid.cpu().numpy(), cmap='jet')
    # plt.axis('off')
    # plt.show()
