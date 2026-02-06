import torch
import torch.nn as nn
import numpy as np
import cv2
import torch.nn.functional as F
import torch
import torch.nn.functional as F


def gaussian_kernel(device, channels=3, kernel_size=5, sigma=1.0):
    """生成可分离高斯卷积核"""
    # 坐标网格
    x_coord = torch.arange(kernel_size, device=device)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    # 计算高斯分布
    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.
    kernel = torch.exp(-torch.sum((xy_grid - mean) ** 2, dim=-1) / (2 * variance))
    kernel = kernel / torch.sum(kernel)
    # 扩展为多通道卷积核
    return kernel[None, None, ...].repeat(channels, 1, 1, 1)


def gaussian_blur(x, kernel):
    """执行高斯模糊"""
    channels = x.shape[1]
    padding = kernel.shape[-1] // 2
    return F.conv2d(x, kernel, padding=padding, groups=channels)


def downsample(x, kernel):
    """高斯下采样"""
    x_blur = gaussian_blur(x, kernel)
    return x_blur[:, :, ::2, ::2]  # 保持相位对齐


def upsample(x, kernel, target_size):
    """高斯上采样"""
    # 双线性上采样
    x_up = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
    # 动态调整奇偶尺寸
    pad_h = target_size[-2] % 2
    pad_w = target_size[-1] % 2
    x_up = F.pad(x_up, (0, pad_w, 0, pad_h))
    return gaussian_blur(x_up, kernel)


def laplacian_pyramid(x: torch.Tensor, levels: int = 4) -> list:
    """
    批量图像拉普拉斯金字塔生成函数
    :param x: 输入张量 [B, C, H, W]
    :param levels: 金字塔层数
    :return: 拉普拉斯金字塔列表，从细到粗
    """
    # 主逻辑
    B, C, H, W = x.shape
    device = x.device
    kernel = gaussian_kernel(device, C, sigma=1.0)
    # 构建高斯金字塔
    gauss_pyr = [x]
    for _ in range(levels - 1):
        gauss_pyr.append(downsample(gauss_pyr[-1], kernel))
    # 构建拉普拉斯金字塔
    lap_pyr = []
    for i in range(levels - 1):
        upsampled = upsample(gauss_pyr[i + 1], kernel, gauss_pyr[i].shape[-2:])
        lap_pyr.append(gauss_pyr[i] - upsampled)
    # 添加最顶层高斯层
    lap_pyr.append(gauss_pyr[-1])
    return lap_pyr


def reconstruct_from_pyramid(lap_pyramid: list) -> torch.Tensor:
    """
    从拉普拉斯金字塔重建原始图像
    :param lap_pyramid: 拉普拉斯金字塔列表（从分解函数输出）
    :return: 重建图像 [B,C,H,W]
    """
    # 初始化重建图像
    recon = lap_pyramid[-1]
    kernel = gaussian_kernel(recon.device, recon.shape[1])

    # 从顶层向底层重建
    for lap_layer in reversed(lap_pyramid[:-1]):
        # 上采样当前重建结果
        upsampled = upsample(recon, kernel, lap_layer.shape[-2:])
        # 叠加拉普拉斯细节
        recon = upsampled + lap_layer
    return recon

"""
    第一个创新点初版代码，弃置，上面函数有引用
"""

class EnhanceModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.low = LowFreqPart()
        self.high = HighFreqPart()

    def forward(self, img):
        h1, h2, h3, ll = laplacian_pyramid(img)
        ll = self.low(img)

        k = gaussian_kernel(ll.device)
        for _ in range(3):
            ll = downsample(ll, k)

        hh1, hh2, hh3 = self.high(h1, h2, h3)
        # return reconstruct_from_pyramid([hh1, hh2, hh3, ll])
        return reconstruct_from_pyramid([hh1, hh2, hh3, ll]) + img


def batch_channel_bilateral_filter(
        x: torch.Tensor,  # 输入张量 (B, C, H, W)
        sigma_space: torch.Tensor,  # 空间参数 (B, C, 1, 1)
        sigma_color: torch.Tensor,  # 颜色参数 (B, C, 1, 1)
        kernel_size: int = 5
) -> torch.Tensor:
    """
    支持Batch-Channel独立参数的双边滤波
    时间复杂度: O(B*C*H*W*K^2)
    内存消耗: 约 B*C*K^2*H*W
    """
    # 输入验证
    assert x.dim() == 4, "输入必须是4D张量 (B, C, H, W)"
    B, C, H, W = x.shape
    assert sigma_space.shape == (B, C, 1, 1), f"空间参数形状应为({B}, {C}, 1, 1)，实际是{sigma_space.shape}"
    assert sigma_color.shape == (B, C, 1, 1), f"颜色参数形状应为({B}, {C}, 1, 1)，实际是{sigma_color.shape}"
    # 生成坐标网格
    radius = (kernel_size - 1) // 2
    dx = torch.arange(-radius, radius + 1, device=x.device)
    dy = torch.arange(-radius, radius + 1, device=x.device)
    dx, dy = torch.meshgrid(dx, dy, indexing='ij')
    d_sq = dx ** 2 + dy ** 2  # (k, k)
    # 计算空间权重 (B, C, k*k)
    spatial_weights = torch.exp(
        -d_sq.flatten()[None, None, :] /  # 添加两个维度用于广播
        (2 * sigma_space.squeeze(-1).squeeze(-1)[..., None] ** 2)  # (B, C, 1)
    )  # -> (B, C, k*k)
    # 展开邻域像素
    x_pad = F.pad(x, (radius,) * 4, mode='replicate')
    x_unfold = F.unfold(x_pad, kernel_size=kernel_size)  # (B, C*k*k, H*W)
    x_unfold = x_unfold.view(B, C, -1, H, W)  # (B, C, k*k, H, W)
    # 计算颜色权重
    center = kernel_size ** 2 // 2
    x_center = x_unfold[:, :, [center], :, :]  # (B, C, 1, H, W)
    color_diff_sq = (x_unfold - x_center) ** 2
    color_weights = torch.exp(
        -color_diff_sq /
        (2 * sigma_color.view(B, C, 1, 1, 1) ** 2 + 1e-6)
    )  # (B, C, k*k, H, W)
    # 融合权重并归一化
    total_weights = spatial_weights.view(B, C, -1, 1, 1) * color_weights  # (B, C, k*k, H, W)
    weighted_sum = (x_unfold * total_weights).sum(dim=2)  # (B, C, H, W)
    norm = total_weights.sum(dim=2).clamp(min=1e-6)
    return weighted_sum / norm


class LowFreqPart(nn.Module):
    def __init__(self):
        super(LowFreqPart, self).__init__()
        self.opp = OPP()
        self.local_gamma = LocalGamma()

    def forward(self, x):
        # params: [batch, 8, 1, 1]
        if x.max() > 1:
            x /= 255
        params = self.opp.forward(x)
        # 调整曝光度
        exposure = params[:, 0, :, :].unsqueeze(2).expand(-1, 3, 256, 192)
        exposed_img = x * exposure

        # torch.autograd.set_detect_anomaly(True)
        # 调整整体伽马
        # global_gamma = params[:, 1, :, :].unsqueeze(2).expand(-1, 3, 256, 192)
        # ln_exposed = torch.log(exposed_img)
        # exposed_img = exposed_img + 1e-6 - exposed_img.min()
        # global_gamma_img = torch.pow(exposed_img, global_gamma)
        # global_gamma_img = exposed_img * (1 + global_gamma * ln_exposed + (global_gamma ** 2 / 2) * ln_exposed ** 2)
        global_gamma_img = exposed_img
        # 局部伽马
        local_g = self.local_gamma.forward(global_gamma_img)
        gamma_img = exposed_img * local_g

        smooth_img = batch_channel_bilateral_filter(
            gamma_img,
            params[:, 2:5, :, :],
            params[:, 5:8, :, :]
        )
        return smooth_img


class OPP(nn.Module):
    """
        Optimal Parameters Predictor
    """

    def __init__(self, in_channels=3, final_out=8):
        super(OPP, self).__init__()
        # 3 256, 192-> 32 128 96
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32)
        )
        # 32 128 96 -> 64 64 48
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64)
        )
        # 64 64 48 -> 128 32 24
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128)
        )
        # 128 32 24 -> 128 16 12
        self.layer4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)
        # 128 16 12 -> 128 8 6
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.Dropout2d()
        )
        # 128 8 6 -> final_out 1 1
        self.layer6 = nn.Conv2d(in_channels=128, out_channels=final_out, kernel_size=(8, 6), stride=1, padding=0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x


class LocalGamma(nn.Module):
    def __init__(self, in_channels=3):
        super(LocalGamma, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, stride=1)
        self.conv5 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, stride=1)
        self.conv6 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        a = self.conv1(x)
        b = self.conv2(a)
        c = self.conv3(b)

        d = self.conv4(c) + c
        e = self.conv5(d) + b
        f = self.conv6(e) + a

        return torch.sigmoid(f)


class HighFreqPart(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mld1 = MLD()
        self.mld2 = MLD()
        self.mld3 = MLD()
        self.csf = CrossScaleFusion()

    def forward(self, x1, x2, x3):
        y1 = self.mld1(x1)
        y2 = self.mld2(x2)
        y3 = self.mld1(x3)
        return self.csf(y1, y2, y3)


class MLD(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=3, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        # b3hw->b24hw
        x = self.layer1(x)
        # u: b3hw  f:b24hw
        u = self.layer2(x)
        f = self.layer3(x)
        u = u.flatten(start_dim=2)  # b3(hw)
        f = f.flatten(start_dim=2)  # b24(hw)

        v = torch.matmul(u, f.permute(0, 2, 1))  # b 3 24
        result = torch.matmul(v.permute(0, 2, 1), u)  # b 24 (hw)
        result = result.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        result = self.layer4(result)
        return result  # b3hw


class CrossScaleFusion(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(in_channels=9, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=9, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=9, out_channels=3, kernel_size=3, stride=1, padding=1)

    def forward(self, x1, x2, x3):
        x1_2 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x1_3 = F.interpolate(x1, size=x3.shape[2:], mode='bilinear', align_corners=False)

        x2_1 = F.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x2_3 = F.interpolate(x2, size=x3.shape[2:], mode='bilinear', align_corners=False)

        x3_1 = F.interpolate(x3, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x3_2 = F.interpolate(x3, size=x2.shape[2:], mode='bilinear', align_corners=False)

        o_1 = self.conv1(torch.cat((x1, x2_1, x3_1), dim=1))
        o_2 = self.conv1(torch.cat((x2, x1_2, x3_2), dim=1))
        o_3 = self.conv1(torch.cat((x3, x2_3, x1_3), dim=1))
        return o_1, o_2, o_3
