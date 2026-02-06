import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_

conv_kernels = [[2, 4, 8, 16],
                [2, 4, 8],
                [2, 4]]


class FeatureEnhance(nn.Module):
    def __init__(self, in_res=[256, 192], embed_dim=8, layer_channel=2):
        super().__init__()
        self.in_res = in_res
        self.layer_channel = layer_channel

        feat_res = in_res
        in_channel = layer_channel
        out_channel = embed_dim
        self.csam1 = CrossScaleAttentionModule(in_channel=in_channel, img_size=feat_res, embed_dim=out_channel,
                                               patch_size=conv_kernels[0])
        self.conv1 = nn.Conv2d(out_channel + layer_channel, layer_channel, (1, 1), 1)

        feat_res = [int(x / 2) for x in feat_res]
        in_channel = out_channel
        out_channel *= 2
        self.csam2 = CrossScaleAttentionModule(in_channel=in_channel, img_size=feat_res, embed_dim=out_channel,
                                               patch_size=conv_kernels[1])
        self.conv2 = nn.Conv2d(out_channel + layer_channel, layer_channel, (1, 1), 1)

        feat_res = [int(x / 2) for x in feat_res]
        in_channel = out_channel
        out_channel *= 2
        self.csam3 = CrossScaleAttentionModule(in_channel=in_channel, img_size=feat_res, embed_dim=out_channel,
                                               patch_size=conv_kernels[2])
        self.conv3 = nn.Conv2d(out_channel + layer_channel, layer_channel, (1, 1), 1)
        self.conv0 = nn.Conv2d(7 * embed_dim + layer_channel, layer_channel, (1, 1), 1)

    def forward(self, pyd):
        '''
            pyd = [(256,192), (128,96), (64,48), (32,24)]
        '''
        pyd0, pyd1, pyd2, pyd3 = pyd
        fusion1 = self.csam1(pyd0)
        fusion2 = self.csam2(fusion1)
        fusion3 = self.csam3(fusion2)

        fusion0 = torch.cat([F.interpolate(input=fusion1, scale_factor=2, mode="bilinear", align_corners=False),
                             F.interpolate(input=fusion2, scale_factor=4, mode="bilinear", align_corners=False),
                             F.interpolate(input=fusion3, scale_factor=8, mode="bilinear", align_corners=False),
                             pyd0],
                            dim=1)
        fusion1 = torch.cat([fusion1, pyd1], dim=1)
        fusion2 = torch.cat([fusion2, pyd2], dim=1)
        fusion3 = torch.cat([fusion3, pyd3], dim=1)

        out0 = self.conv0(fusion0)
        out1 = self.conv1(fusion1)
        out2 = self.conv2(fusion2)
        out3 = self.conv3(fusion3)
        return [out0, out1, out2, out3]


class CrossScaleAttentionModule(nn.Module):
    def __init__(self, in_channel, img_size, embed_dim, patch_size):
        super().__init__()
        self.in_channel = in_channel
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_channel,
                                      embed_dim=embed_dim)
        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        trunc_normal_(self.absolute_pos_embed, std=.02)
        self.atten = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=1)

    def forward(self, in_feature):
        x = self.patch_embed(in_feature)
        x = x + self.absolute_pos_embed
        x, _ = self.atten(x, x, x)
        x = x.view(x.size(0), int(self.img_size[0] / 2), int(self.img_size[1] / 2), x.size(2))
        x = x.permute(0, 3, 1, 2)
        return x


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: [4].
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=[224, 224], patch_size=[4], in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        # patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[0]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.projs = nn.ModuleList()
        for i, ps in enumerate(patch_size):
            if i == len(patch_size) - 1:
                dim = embed_dim // 2 ** i
            else:
                dim = embed_dim // 2 ** (i + 1)
            stride = patch_size[0]
            padding = (ps - patch_size[0]) // 2
            self.projs.append(nn.Conv2d(in_chans, dim, kernel_size=ps, stride=stride, padding=padding))
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        xs = []
        for i in range(len(self.projs)):
            tx = self.projs[i](x).flatten(2).transpose(1, 2)
            xs.append(tx)  # B Ph*Pw C
        x = torch.cat(xs, dim=2)
        if self.norm is not None:
            x = self.norm(x)
        return x
