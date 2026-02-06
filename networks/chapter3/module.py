import torch
import torch.nn as nn
import numpy as np
import cv2
import torch.nn.functional as F
from networks.chapter3.hvi_trans import RGB_HVI
from networks.Enhance import gaussian_kernel, gaussian_blur, reconstruct_from_pyramid, downsample, upsample
from networks.chapter3.feature_enhance import FeatureEnhance
from networks.chapter3.light_improve import LightImprove
from networks.chapter3.low_rank_denoise import LowRandDenoise

'''
    Multi-Stream Enhance Module
'''


def gaussian_pyramids(x, kernel, levels=4):
    B, C, H, W = x.shape
    device = x.device
    kernel = kernel
    # 构建高斯金字塔
    gauss_pyr = [x]
    for _ in range(levels - 1):
        gauss_pyr.append(downsample(gauss_pyr[-1], kernel))
    return gauss_pyr


def laplacian_pyramid(gaussian_pyd, kernel):
    levels = len(gaussian_pyd)
    lap_pyr = []
    for i in range(levels - 1):
        upsampled = upsample(gaussian_pyd[i + 1], kernel, gaussian_pyd[i].shape[-2:])
        lap_pyr.append(gaussian_pyd[i] - upsampled)
    # 添加最顶层高斯层
    lap_pyr.append(gaussian_pyd[-1])
    return lap_pyr


class MStream(nn.Module):
    def __init__(self, pyd_levels=4, active=[True, True, True]):
        """
        active=[True, True, True]: 光照增强，去噪，特征增强
        """
        super().__init__()
        self.pyd_levels = pyd_levels
        self.hvi = RGB_HVI()
        self.pyd_layer_channels = 2
        self.feat_enhance = FeatureEnhance()
        self.denoise = LowRandDenoise()
        self.light_improve = LightImprove()

        self.active_light = active[0]
        self.active_denoise = active[1]
        self.active_feat_en = active[2]

        self.reconstruct_fusion = nn.ModuleList()
        for i in range(self.pyd_levels - 1):
            conv = nn.Conv2d(in_channels=2 * self.pyd_layer_channels, out_channels=self.pyd_layer_channels,
                             kernel_size=3, padding=1, stride=1)
            self.reconstruct_fusion.append(conv)

    def forward(self, low_light_img):
        H, V, I = self.hvi.HVIT(low_light_img)
        hv = torch.cat([H, V], dim=1)

        B, C, h, w = hv.shape
        device = hv.device
        kernel = gaussian_kernel(device, C, sigma=1.0)
        g_pyd = gaussian_pyramids(hv, kernel)
        l_pyd = laplacian_pyramid(g_pyd, kernel)

        if self.active_feat_en:
            enhanced_g = self.feat_enhance(g_pyd)
        else:
            enhanced_g = g_pyd

        if self.active_denoise:
            noiseless_l = self.denoise(l_pyd)
        else:
            noiseless_l = l_pyd

        noise_map = l_pyd[0] - noiseless_l[0]

        if self.active_light:
            lightened_i = self.light_improve(I, noise_map)
        else:
            lightened_i = I

        # 高斯金字塔+拉普拉斯金字塔 重建
        current = enhanced_g[-1]
        for i in reversed(range(len(enhanced_g) - 1)):
            # 上采样到上一层尺寸
            target_shape = enhanced_g[i].shape[-2:]
            current_up = F.interpolate(current, size=target_shape,
                                       mode='bilinear', align_corners=False)

            # 叠加拉普拉斯细节 (l_pyramid与高斯金字塔索引差1)
            x = current_up + noiseless_l[i]
            x = torch.cat([x, enhanced_g[i]], dim=1)

            # 融合x和当前层高斯特征
            current = self.reconstruct_fusion[i](x)

        H, V = current[:, 0, :, :].squeeze(1), current[:, 1, :, :].squeeze(1)
        lightened_i = lightened_i.squeeze(1)
        # HVI色彩空间重建
        return self.hvi.PHVIT(H, V, lightened_i) + low_light_img
