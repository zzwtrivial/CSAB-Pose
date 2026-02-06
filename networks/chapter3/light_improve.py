import torch
import torch.nn as nn
from timm.layers import trunc_normal_
import torch.nn.functional as F


class LightImprove(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 patch_size: int = 16,
                 num_heads: int = 8,
                 img_size: list = [256, 192]):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = patch_size * patch_size * in_channels

        # n的通道数是i的两倍
        self.i_pos_embed = nn.Parameter(
            torch.zeros(1, int(img_size[0] / patch_size * img_size[1] / patch_size), self.embed_dim))
        trunc_normal_(self.i_pos_embed, std=.02)
        self.n_pos_embed = nn.Parameter(
            torch.zeros(1, int(img_size[0] / patch_size * img_size[1] / patch_size), 2 * self.embed_dim))
        trunc_normal_(self.n_pos_embed, std=.02)

        # 投影层（生成 Q/K/V）
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.kv_proj = nn.Linear(2 * self.embed_dim, 2 * self.embed_dim)

        # 多头注意力
        self.attn = nn.MultiheadAttention(self.embed_dim, num_heads, batch_first=True)

    def forward(self, I: torch.Tensor, N: torch.Tensor) -> torch.Tensor:
        """
        Args:
            I (Tensor): 查询特征图，形状 [B, C, H, W]
            N (Tensor): 键值特征图，形状 [B, C, H, W]
        Returns:
            output (Tensor): 注意力输出，形状 [B, C, H, W]
        """

        B, C, H, W = I.shape
        p = self.patch_size

        # --- Step 1: 分块处理 ---
        # 将特征图分割为 p x p 的块
        def to_patches(x):
            _, c, _, _ = x.shape
            x = x.unfold(2, p, p).unfold(3, p, p)  # [B, C, H//p, W//p, p, p]
            x = x.permute(0, 2, 3, 1, 4, 5)  # [B, H//p, W//p, C, p, p]
            x = x.reshape(B, -1, c * p ** 2)  # [B, (H//p * W//p), C*p²]
            return x

        I_patches = to_patches(I)  # [B, n_patches, C*p²]
        N_patches = to_patches(N)

        # 添加到分块后的特征
        I_patches = I_patches + self.i_pos_embed
        N_patches = N_patches + self.n_pos_embed

        # --- Step 3: 投影到 Q/K/V 空间 ---
        Q = self.q_proj(I_patches)  # [B, n_patches, embed_dim]
        K, V = self.kv_proj(N_patches).chunk(2, dim=-1)

        # --- Step 4: 交叉注意力计算 ---
        attn_output, _ = self.attn(Q, K, V)  # [B, n_patches, embed_dim]

        # --- Step 5: 恢复空间维度 ---
        # 将输出转换为 [B, C, H, W]
        output = attn_output.view(B, H // p, W // p, C, p, p)
        output = output.permute(0, 3, 1, 4, 2, 5)  # [B, C, H//p, p, W//p, p]
        output = output.contiguous().view(B, C, H, W)
        output = F.sigmoid(output)

        return output * I
