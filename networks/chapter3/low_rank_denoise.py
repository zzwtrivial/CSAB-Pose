import torch
import torch.nn as nn
import numpy as np
import cv2
import torch.nn.functional as F
import torch
import torch.nn.functional as F


class LowRandDenoise(nn.Module):
    def __init__(self):
        super().__init__()
        self.mld1 = MLD()
        self.mld2 = MLD()
        self.mld3 = MLD()
        self.csf = CrossScaleFusion()

    def forward(self, pyd):
        x1, x2, x3, x4 = pyd
        y1 = self.mld1(x1)
        y2 = self.mld2(x2)
        y3 = self.mld1(x3)
        o1, o2, o3 = self.csf(y1, y2, y3)
        return [o1, o2, o3, x4]


class MLD(nn.Module):
    def __init__(self, in_channels=2, mid_channels=24):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=mid_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=mid_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
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
    def __init__(self, in_channels=2, in_nums=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels * in_nums, out_channels=in_channels, kernel_size=3, stride=1,
                               padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels * in_nums, out_channels=in_channels, kernel_size=3, stride=1,
                               padding=1)
        self.conv3 = nn.Conv2d(in_channels=in_channels * in_nums, out_channels=in_channels, kernel_size=3, stride=1,
                               padding=1)

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
