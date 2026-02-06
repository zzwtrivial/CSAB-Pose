import sys

from networks.other.CIDNet.CIDNet import CIDNet
from networks.resnet import *
import torch.nn as nn
import torch
from networks.globalNet import globalNet
from networks.refineNet import refineNet
from networks.feat_enhancer import FeatEnHancer
from networks.chapter3.module import MStream
from networks.other.lee.resnetlsbn import resnet50lsbn
from networks.other.LLFlow.LLFlow_model import get_model as get_llflow
from networks.other.LLFlow.LLFlow_model import auto_padding, hiseq_color_cv2_img
from networks.resnet_BYOL import ResNet50_BYOL
import numpy as np
from networks.other.lime.exposure_enhancement import enhance_image_exposure

__all__ = ['CPN50', 'CPN101', 'E_CPN50', 'FeatEn_CPN50', 'LSBN_CPN50', "LLFlow_CPN50", "LIME_CPN50",
           "CIConv_CPN50", "Chapter4_model"]

from networks.resnet import resnet50_ciconv


class CPN(nn.Module):
    def __init__(self, resnet, output_shape, num_class, pretrained=True):
        super(CPN, self).__init__()
        channel_settings = [2048, 1024, 512, 256]
        self.resnet = resnet
        self.global_net = globalNet(channel_settings, output_shape, num_class)
        self.refine_net = refineNet(channel_settings[-1], output_shape, num_class)

    def forward(self, x):
        res_out = self.resnet(x)
        global_fms, global_outs = self.global_net(res_out)
        refine_out = self.refine_net(global_fms)

        return global_outs, refine_out


class FeatEnCPN(nn.Module):
    def __init__(self, cpn, freeze: bool):
        super().__init__()
        self.cpn = cpn
        self.fe = FeatEnHancer()
        if freeze:
            for p in self.cpn.parameters():
                p.requires_grad = False

    def forward(self, x):
        x = self.fe(x)
        global_outs, refine_out = self.cpn(x)
        return global_outs, refine_out


class CID_CPN(nn.Module):
    def __init__(self, cpn, cid_path):
        super().__init__()
        self.cid = CIDNet()
        self.cid.load_state_dict(torch.load(cid_path, map_location=lambda storage, loc: storage))
        for p in self.cid.parameters():
            p.requires_grad = False
        self.cpn = cpn

    def forward(self, x):
        x = self.cid(x)
        global_outs, refine_out = self.cpn(x)
        return global_outs, refine_out


class EnhanceCPN(nn.Module):
    def __init__(self, cpn, freeze, active=[True, True, True]):
        super().__init__()
        self.cpn = cpn
        self.enhance = MStream(active=active)
        if freeze:
            for p in self.cpn.parameters():
                p.requires_grad = False

    def forward(self, x):
        # x *= 255
        enhance_x = self.enhance(x)
        # if enhance_x.max() > 1:
        #     enhance_x /= 255
        global_outs, refine_out = self.cpn(enhance_x)
        return global_outs, refine_out, enhance_x


class CPN_lsbn(nn.Module):
    def __init__(self, resnet_lsbn, output_shape, num_class, pretrained=True):
        super(CPN_lsbn, self).__init__()
        channel_settings = [2048, 1024, 512, 256]
        self.resnet_lsbn = resnet_lsbn
        self.global_net = globalNet(channel_settings, output_shape, num_class)
        self.refine_net = refineNet(channel_settings[-1], output_shape, num_class)

    def forward(self, x, y):
        x0, x1, x2, x3, x4 = self.resnet_lsbn(x, y)
        res_out = [x4, x3, x2, x1]
        global_fms, global_outs = self.global_net(res_out)
        refine_out = self.refine_net(global_fms)

        return x0, x1, x2, x3, x4, global_outs, refine_out


class LLFlow_CPN(nn.Module):
    def __init__(self, cpn):
        super().__init__()
        self.cpn = cpn
        self.llflow = get_llflow()
        for p in self.llflow.netG.parameters():
            p.requires_grad = False
            p.data = p.data.to(torch.float32)

    def forward(self, x):
        device = x.device
        d_x = x.cpu().detach()
        xx = []
        for img_chw in d_x:
            img_hwc = img_chw.permute(1, 2, 0).numpy()
            img_hwc = (img_hwc * 255).clip(0, 255).astype(np.uint8)

            lr, padding_params = auto_padding(img_hwc)
            his = hiseq_color_cv2_img(lr)
            lr_t = torch.Tensor(np.expand_dims(lr.transpose([2, 0, 1]), axis=0).astype(np.float32)) / 255
            lr_t = torch.log(torch.clamp(lr_t + 1e-3, min=1e-3))
            his = torch.Tensor(np.expand_dims(his.transpose([2, 0, 1]), axis=0).astype(np.float32)) / 255
            lr_t = torch.cat([lr_t, his], dim=1)

            sr_t = self.llflow.get_sr(lq=lr_t.cuda(), heat=None)

            def rgb(t): return (
                    np.clip((t[0] if len(t.shape) == 4 else t).detach().cpu().numpy().transpose([1, 2, 0]), 0,
                            1) * 255).astype(
                np.uint8)

            sr = rgb(torch.clamp(sr_t, 0, 1)[:, :, padding_params[0]:sr_t.shape[2] - padding_params[1],
                     padding_params[2]:sr_t.shape[3] - padding_params[3]])
            xx.append(sr)
        chw_list = [img.transpose(2, 0, 1) for img in xx]
        batch_np = np.stack(chw_list, axis=0)  # 形状 (B, 3, 256, 192)
        tensor = torch.from_numpy(batch_np).to(device)

        global_outs, refine_out = self.cpn(tensor)
        return global_outs, refine_out


class LIME_CPN(nn.Module):
    def __init__(self, cpn, gamma: float = 0.6, lambda_: float = 0.15, dual: bool = False, sigma: int = 3,
                 bc: float = 1, bs: float = 1, be: float = 1, eps: float = 1e-3):
        super().__init__()
        self.cpn = cpn
        self.gamma = gamma
        self.lamda = lambda_
        self.dual = dual
        self.sigma = sigma
        self.bc = bc
        self.bs = bs
        self.be = be
        self.eps = eps

    def forward(self, x):
        # 进来的x [0,1]
        device = x.device
        d_x = x.cpu().detach()
        xx = []
        for img_chw in d_x:
            img_hwc = img_chw.permute(1, 2, 0).numpy()
            img_hwc = (img_hwc * 255).clip(0, 255).astype(np.uint8)
            img_hwc = enhance_image_exposure(img_hwc, self.gamma, self.lamda, self.dual, self.sigma, self.bc, self.bs,
                                             self.be, self.eps)
            xx.append(img_hwc)
        xx = [img.transpose(2, 0, 1) for img in xx]
        batch_np = np.stack(xx, axis=0)  # 形状 (B, 3, 256, 192)
        tensor = torch.from_numpy(batch_np).to(device)
        global_outs, refine_out = self.cpn(tensor / 255)
        return global_outs, refine_out


class CPN50_BYOL(nn.Module):
    def __init__(self, resnet_BYOL: ResNet50_BYOL, output_shape, num_class, pretrained=True):
        super(CPN50_BYOL, self).__init__()
        channel_settings = [2048, 1024, 512, 256]
        self.resnet = resnet_BYOL
        self.global_net = globalNet(channel_settings, output_shape, num_class)
        self.refine_net = refineNet(channel_settings[-1], output_shape, num_class)

    def forward(self, night, day=None, dual=False, train_head=False, update=True, similarity=False, proj=False):
        if dual:
            outputs_night, outputs_day, feat_q, feat_k, feat_q2, feat_k2 = \
                self.resnet(x=night, y=day, dual=dual, train_head=train_head, update=update, similarity=similarity,
                            proj=proj)
        else:
            outputs_night = self.resnet(x=night, y=day, dual=dual, train_head=train_head, update=update,
                                        similarity=similarity,
                                        proj=proj)
        global_fms, global_outs = self.global_net(outputs_night)
        refine_out = self.refine_net(global_fms)

        if dual:
            return global_outs, refine_out, feat_q, feat_k, feat_q2, feat_k2
        else:
            return global_outs, refine_out


def CPN50(out_size, num_class, freeze: bool = False, pretrained=True):
    res50 = resnet50(pretrained=pretrained)
    model = CPN(res50, output_shape=out_size, num_class=num_class, pretrained=pretrained)
    return model


def CPN101(out_size, num_class, freeze: bool = False, pretrained=True):
    res101 = resnet101(pretrained=pretrained)
    model = CPN(res101, output_shape=out_size, num_class=num_class, pretrained=pretrained)
    return model


def E_CPN50(out_size, num_class, freeze: bool = False, pretrained=True, active=[True, True, True]):
    """
    active=[True, True, True]: 光照增强，去噪，特征增强
    """
    cpn = CPN50(out_size, num_class, pretrained)
    model = EnhanceCPN(cpn, freeze, active=active)
    return model


def FeatEn_CPN50(out_size, num_class, freeze: bool, pretrained=True):
    cpn = CPN50(out_size, num_class, pretrained)
    model = FeatEnCPN(cpn, freeze)
    return model


def LSBN_CPN50(out_size, num_class, in_features=0, num_conditions=2, pretrained=True):
    res50 = resnet50lsbn(pretrained=pretrained, num_class=num_class, in_features=in_features,
                         num_conditions=num_conditions)
    model = CPN_lsbn(res50, output_shape=out_size, num_class=num_class, pretrained=pretrained)
    return model


def LLFlow_CPN50(out_size, num_class, freeze: bool, pretrained=True):
    cpn = CPN50(out_size, num_class, pretrained)
    model = LLFlow_CPN(cpn)
    return model


def LIME_CPN50(out_size, num_class, freeze: bool = False, pretrained=True):
    cpn = CPN50(out_size, num_class, pretrained)
    model = LIME_CPN(cpn)
    return model


def CID_CPN50(out_size, num_class, cid_path, freeze: bool = False, pretrained=True):
    cpn = CPN50(out_size, num_class, pretrained)
    model = CID_CPN(cpn, cid_path)
    return model


def CIConv_CPN50(out_size, num_class, pretrained=True):
    res50 = resnet50_ciconv(pretrained=pretrained, invariant='W', k=3, scale=0.01)
    cpn = CPN(res50, output_shape=out_size, num_class=num_class, pretrained=pretrained)
    return cpn


def Chapter4_model(out_size, num_class, pretrained=True):
    res50 = resnet50(pretrained=True)
    res50_byol = ResNet50_BYOL(res50)
    model = CPN(res50, out_size, num_class, pretrained)
    return model
