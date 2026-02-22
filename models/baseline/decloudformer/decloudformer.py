# https://github.com/N1rv4n4/DecloudFormer/blob/main/models/decloudformer.py
"""没仔细看，几乎照搬，只改了attention mask机制用于云去除"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import to_2tuple, trunc_normal_


class RLN(nn.Module):
    r"""Revised LayerNorm"""

    def __init__(self, dim, eps=1e-5, detach_grad=False):
        super(RLN, self).__init__()
        self.eps = eps
        self.detach_grad = detach_grad

        self.weight = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.bias = nn.Parameter(torch.zeros((1, dim, 1, 1)))

        self.meta1 = nn.Conv2d(1, dim, 1)
        self.meta2 = nn.Conv2d(1, dim, 1)

        trunc_normal_(self.meta1.weight, std=.02)  # 正态分布截断
        nn.init.constant_(self.meta1.bias, 1)  # 将bias初始化为1

        trunc_normal_(self.meta2.weight, std=.02)  # 正态分布截断
        nn.init.constant_(self.meta2.bias, 0)  # 将bias初始化为1

    def forward(self, input):
        mean = torch.mean(input, dim=(1, 2, 3), keepdim=True)
        std = torch.sqrt((input - mean).pow(2).mean(dim=(1, 2, 3), keepdim=True) + self.eps)

        normalized_input = (input - mean) / std

        if self.detach_grad:
            rescale, rebias = self.meta1(std.detach()), self.meta2(mean.detach())
        else:
            rescale, rebias = self.meta1(std), self.meta2(mean)

        out = normalized_input * self.weight + self.bias
        return out, rescale, rebias


class Mlp(nn.Module):
    def __init__(self, network_depth, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.network_depth = network_depth

        self.mlp = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_features, out_features, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            gain = (8 * self.network_depth) ** (-1 / 4)
            fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size ** 2, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def get_relative_positions(window_size):
    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)

    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_positions = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww

    relative_positions = relative_positions.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_positions_log = torch.sign(relative_positions) * torch.log(1. + relative_positions.abs())

    return relative_positions_log


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        relative_positions = get_relative_positions(self.window_size)
        self.register_buffer("relative_positions", relative_positions)
        self.meta = nn.Sequential(
            nn.Linear(2, 256, bias=True),
            nn.ReLU(True),
            nn.Linear(256, num_heads, bias=True)
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv, mask=None):
        B_, N, _ = qkv.shape  # 1024 49 144
        # 3 1024 2 49 24
        qkv = qkv.reshape(B_, N, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)
        # 1024 2 49 24
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q = q * self.scale  # 1024 2 49 24
        attn = (q @ k.transpose(-2, -1))  # 1024 2 49 49

        relative_position_bias = self.meta(self.relative_positions)  # 49 49 2
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww # 2 49 49
        attn = attn + relative_position_bias.unsqueeze(0)  # 1024 2 49 49 + 1 2 49 49

        if mask is not None:  # 1024 49 49
            nW = mask.shape[0]  # 1024
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(
                0)  # 1 1024 2 49 49 + 1 1024 1 49 49 -> 1 1024 2 49 49
            attn = attn.view(-1, self.num_heads, N, N)  # 1 1024 2 49 49 -> 1024 2 49 49
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        # attn = self.softmax(attn)

        x = (attn @ v)  # 1024 2 49 24
        x = x.transpose(1, 2)  # 1024 2 49 24 -> 1024 49 2 24
        x = x.reshape(B_, N, self.dim)  # 1024 49 48
        return x


class Attention(nn.Module):
    def __init__(self, network_depth, dim, input_resolution, num_heads, window_size, shift_size=0, use_attn=False,
                 conv_type=None, ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.head_dim = int(dim // num_heads)
        self.num_heads = num_heads

        self.window_size = window_size
        self.shift_size = shift_size

        self.network_depth = network_depth
        self.use_attn = use_attn
        self.conv_type = conv_type
        if self.input_resolution <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = self.input_resolution

        if self.conv_type == 'Conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.ReLU(True),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect')
            )

        if self.conv_type == 'DWConv':
            self.conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode='reflect')

        if self.conv_type == 'DWConv' or self.use_attn:
            self.V = nn.Conv2d(dim, dim, 1)
            self.proj = nn.Conv2d(dim, dim, 1)

        if self.use_attn:
            self.QK = nn.Conv2d(dim, dim * 2, 1)
            self.attn = WindowAttention(dim, window_size, num_heads)
            self.cloud_conv = nn.Conv2d(dim * 2, dim, 1)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H = self.input_resolution
            W = self.input_resolution
            img_mask = torch.zeros((1, int(H), int(W), 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            '''
            0 -7 None
            -7 -3 None
            -3 None None
            '''
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            '''
            0 -7 None
            -7 -3 None
            -3 None None
            '''
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1 (1 75 75 1) 5
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            # 1024 49 49
            mask_sum = attn_mask.sum(dim=[1, 2]).numpy().astype(np.int32)
            mask_uq = np.unique(mask_sum)
            # index_1 = np.where(mask_sum == mask_uq[0])
            index_2 = np.where(mask_sum == mask_uq[1])
            # print(index_2[0])
            # index_3 = np.where(mask_sum == mask_uq[2])
            cloud_mask = attn_mask
            cloud_mask[index_2] = -100.0
        # for idx in index_2[0].tolist():
        # 	mask_w = cloud_mask.shape[2]
        # 	cloud_mask_line = cloud_mask[idx,0,:mask_w]
        # 	line_sum = cloud_mask_line.sum()
        # 	if line_sum < 0:
        # 		cloud_mask[idx] = -100.0

        # np.set_printoptions(threshold=np.inf)
        # print(mask_sum.numpy())
        # np.savetxt('/home/server4/lmk/DehazeFormer/attn_mask.txt', attn_mask.numpy().reshape(-1, 49*49), fmt='%f', delimiter=' ')
        # for i in range(1024):
        # 	mask_img = cloud_mask[i].numpy()
        # 	mask_img = mask_img + 101
        # 	mask_img = mask_img.astype(np.uint8)
        # 	cv.imwrite('/home/server4/lmk/DehazeFormer/attn_mask/cloud_{}.png'.format(i), mask_img)

        else:
            attn_mask = None
            cloud_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.register_buffer("cloud_mask", cloud_mask)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            w_shape = m.weight.shape

            if w_shape[0] == self.dim * 2:  # QK
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)
            else:
                gain = (8 * self.network_depth) ** (-1 / 4)
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def check_size(self, x, shift=False):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size

        if shift:
            x = F.pad(x, (self.shift_size, (self.window_size - self.shift_size + mod_pad_w) % self.window_size,
                          self.shift_size, (self.window_size - self.shift_size + mod_pad_h) % self.window_size),
                      mode='reflect')
        else:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, x):
        B, C, H, W = x.shape  # 1 48 224 224
        x = x.permute(0, 2, 3, 1)  # 1 224 224 48

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size),
                                   dims=(1, 2))  # 1 224 224 48 -> # 1 224 224 48
            shifted_x = shifted_x.permute(0, 3, 1, 2)  # 1 48 224 224
        # x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C # 1 56 56 96 -> 64 7 7 96
        else:
            shifted_x = x  # 1 224 224 48
            shifted_x = shifted_x.permute(0, 3, 1, 2)  # 1 48 224 224
        # partition windows
        # x_windows = window_partition(shifted_x, self.window_size)  # 1024 49 48 # nW*B, window_size, window_size, C

        if self.conv_type == 'DWConv' or self.use_attn:
            V = self.V(shifted_x)  # 1 48 224 224

        if self.use_attn:
            QK = self.QK(shifted_x)  # 1 48 224 224 -> 1 96 224 224
            QKV = torch.cat([QK, V], dim=1)  # 1 144 224 224

            # shift
            # shifted_QKV = self.check_size(QKV, self.shift_size > 0) # 1 144 224 224
            # Ht, Wt = shifted_QKV.shape[2:] # 224 224

            # partition windows
            QKV = QKV.permute(0, 2, 3, 1)  # 1 224 224 144
            qkv = window_partition(QKV, self.window_size)  # 1024 49 144 # nW*B, window_size**2, C

            # attn_windows = self.attn(qkv) # 1024 49 144 -> 1024 49 48
            # W-MSA/SW-MSA
            attn_windows_all = self.attn(qkv, mask=self.attn_mask)  # 1024 49 144 -> 1024 49 48
            attn_windows_cloud = self.attn(qkv, mask=self.cloud_mask)  # 1024 49 144 -> 1024 49 48

            attn_windows = torch.cat([attn_windows_all, attn_windows_cloud], dim=2)  # 1024 49 96
            attn_windows = self.cloud_conv(attn_windows.permute(2, 0, 1))  # 1024 49 96 -> 96 1024 49 -> 48 1024 49
            attn_windows = attn_windows.permute(1, 2, 0)  # 48 1024 49 -> 1024 49 48

            # merge windows
            # shifted_out = window_reverse(attn_windows, self.window_size, Ht, Wt)  # 1 224 224 48 B H' W' C #
            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # 1024 49 48 -> 1024 7 7 48

            # reverse cyclic shift
            if self.shift_size > 0:
                shifted_x = window_reverse(attn_windows, self.window_size, H,
                                           W)  # B H' W' C # 1024 7 7 48 -> 1 224 224 48
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))  # 1 224 224 48
            else:
                shifted_x = window_reverse(attn_windows, self.window_size, H,
                                           W)  # B H' W' C # 1024 7 7 48 -> 1 224 224 48
                x = shifted_x  # 1 224 224 48

            # reverse cyclic shift
            # out = shifted_out[:, self.shift_size:(self.shift_size+H), self.shift_size:(self.shift_size+W), :] # 1 224 224 48
            x = x.permute(0, 3, 1, 2)  # 1 48 224 224

            if self.conv_type in ['Conv', 'DWConv']:
                conv_out = self.conv(V)  # 1 48 224 224
                out = self.proj(conv_out + x)  # 1 48 224 224
            else:
                out = self.proj(x)

        else:
            if self.conv_type == 'Conv':
                out = self.conv(shifted_x)  # no attention and use conv, no projection
            elif self.conv_type == 'DWConv':
                out = self.proj(self.conv(V))

        return out


class TransformerBlock(nn.Module):
    def __init__(self, network_depth, dim, input_resolution, num_heads, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, mlp_norm=False,
                 window_size=8, shift_size=0, use_attn=True, conv_type=None):
        super().__init__()
        self.use_attn = use_attn
        self.mlp_norm = mlp_norm

        # self.norm1 = norm_layer(dim) if use_attn else nn.Identity()
        self.norm1 = nn.GroupNorm(4, dim) if use_attn else nn.Identity()
        self.attn = Attention(network_depth, dim, input_resolution=input_resolution, num_heads=num_heads,
                              window_size=window_size,
                              shift_size=shift_size, use_attn=use_attn, conv_type=conv_type)

        # self.norm2 = norm_layer(dim) if use_attn and mlp_norm else nn.Identity()
        self.norm2 = nn.GroupNorm(4, dim) if use_attn and mlp_norm else nn.Identity()
        self.mlp = Mlp(network_depth, dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x):
        identity = x  # 1 48 224 224
        # if self.use_attn: x, rescale, rebias = self.norm1(x) # 1 48 224 224
        # x = self.attn(x) # 1 48 224 224 -> # 1 48 224 224
        # if self.use_attn: x = x * rescale + rebias # 1 48 224 224
        # x = identity + x # 1 48 224 224
        if self.use_attn: x = self.norm1(x)  # 1 48 224 224
        x = self.attn(x)  # 1 48 224 224 -> # 1 48 224 224
        if self.use_attn: x = x  # 1 48 224 224
        x = identity + x  # 1 48 224 224

        identity = x  # 1 48 224 224
        # if self.use_attn and self.mlp_norm: x, rescale, rebias = self.norm2(x) # 1 48 224 224
        # x = self.mlp(x) # 1 48 224 224
        # if self.use_attn and self.mlp_norm: x = x * rescale + rebias # 1 48 224 224
        # x = identity + x # 1 48 224 224
        if self.use_attn and self.mlp_norm: x = self.norm2(x)  # 1 48 224 224
        x = self.mlp(x)  # 1 48 224 224
        if self.use_attn and self.mlp_norm: x = x  # 1 48 224 224
        x = identity + x  # 1 48 224 224
        return x


class BasicLayer(nn.Module):
    def __init__(self, network_depth, dim, input_resolution, depth, num_heads, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, window_size=8,
                 attn_ratio=0., attn_loc='last', conv_type=None):

        super().__init__()
        self.dim = dim
        self.depth = depth

        attn_depth = attn_ratio * depth

        if attn_loc == 'last':
            use_attns = [i >= depth - attn_depth for i in range(depth)]
        elif attn_loc == 'first':
            use_attns = [i < attn_depth for i in range(depth)]
        elif attn_loc == 'middle':
            use_attns = [i >= (depth - attn_depth) // 2 and i < (depth + attn_depth) // 2 for i in range(depth)]

        # build blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(network_depth=network_depth,
                             dim=dim,
                             input_resolution=input_resolution,
                             num_heads=num_heads,
                             mlp_ratio=mlp_ratio,
                             norm_layer=norm_layer,
                             window_size=window_size,
                             shift_size=0 if (i % 2 == 0) else window_size // 2,
                             use_attn=use_attns[i], conv_type=conv_type)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:  # 1 48 224 224
            x = blk(x)
        return x  # 1 48 224 224


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=4, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=4, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(SKFusion, self).__init__()

        self.height = height
        d = max(int(dim / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out


class DecloudFormer(nn.Module):
    def __init__(self, in_chans=3, out_chans=4, window_size=8,
                 embed_dims=[24, 48, 96, 48, 24],
                 mlp_ratios=[2., 4., 4., 2., 2.],
                 depths=[16, 16, 16, 8, 8],
                 num_heads=[2, 4, 6, 1, 1],
                 attn_ratio=[1 / 4, 1 / 2, 3 / 4, 0, 0],
                 conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'],
                 norm_layer=[RLN, RLN, RLN, RLN, RLN]):
        super(DecloudFormer, self).__init__()

        # setting
        self.patch_size = 4
        self.window_size = window_size
        self.mlp_ratios = mlp_ratios
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.input_size = 224

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)

        # backbone
        self.layer1 = BasicLayer(network_depth=sum(depths), dim=embed_dims[0], input_resolution=self.input_size,
                                 depth=depths[0],
                                 num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
                                 norm_layer=norm_layer[0], window_size=window_size,
                                 attn_ratio=attn_ratio[0], attn_loc='last', conv_type=conv_type[0])

        self.patch_merge1 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])

        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)

        self.layer2 = BasicLayer(network_depth=sum(depths), dim=embed_dims[1], input_resolution=self.input_size / 2,
                                 depth=depths[1],
                                 num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
                                 norm_layer=norm_layer[1], window_size=window_size,
                                 attn_ratio=attn_ratio[1], attn_loc='last', conv_type=conv_type[1])

        self.patch_merge2 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])

        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)

        self.layer3 = BasicLayer(network_depth=sum(depths), dim=embed_dims[2], input_resolution=self.input_size / 4,
                                 depth=depths[2],
                                 num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
                                 norm_layer=norm_layer[2], window_size=window_size,
                                 attn_ratio=attn_ratio[2], attn_loc='last', conv_type=conv_type[2])

        self.patch_split1 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[3], embed_dim=embed_dims[2])

        assert embed_dims[1] == embed_dims[3]
        self.fusion1 = SKFusion(embed_dims[3])

        self.layer4 = BasicLayer(network_depth=sum(depths), dim=embed_dims[3], input_resolution=self.input_size / 2,
                                 depth=depths[3],
                                 num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
                                 norm_layer=norm_layer[3], window_size=window_size,
                                 attn_ratio=attn_ratio[3], attn_loc='last', conv_type=conv_type[3])

        self.patch_split2 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])

        assert embed_dims[0] == embed_dims[4]
        self.fusion2 = SKFusion(embed_dims[4])

        self.layer5 = BasicLayer(network_depth=sum(depths), dim=embed_dims[4], input_resolution=self.input_size,
                                 depth=depths[4],
                                 num_heads=num_heads[4], mlp_ratio=mlp_ratios[4],
                                 norm_layer=norm_layer[4], window_size=window_size,
                                 attn_ratio=attn_ratio[4], attn_loc='last', conv_type=conv_type[4])

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            patch_size=1, out_chans=out_chans, embed_dim=embed_dims[4], kernel_size=3)

    def check_image_size(self, x):
        # NOTE: for I2I test
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        x = self.patch_embed(x)  # 1 3 224 224 -> 1 48 224 224
        x = self.layer1(x)
        skip1 = x

        x = self.patch_merge1(x)
        x = self.layer2(x)
        skip2 = x

        x = self.patch_merge2(x)
        x = self.layer3(x)
        x = self.patch_split1(x)

        x = self.fusion1([x, self.skip2(skip2)]) + x
        x = self.layer4(x)
        x = self.patch_split2(x)

        x = self.fusion2([x, self.skip1(skip1)]) + x
        x = self.layer5(x)
        x = self.patch_unembed(x)
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)  # 1 3 224 224

        feat = self.forward_features(x)
        K, B = torch.split(feat, (1, self.in_chans), dim=1)  # 1 4 224 224 -> 1 1 224 224 & 1 3 224 224

        x = K * x - B + x  # 1 3 224 224
        x = x[:, :, :H, :W]
        return x


def decloudformer_t(in_chans=3, out_chans=4, ):
    return DecloudFormer(in_chans=in_chans, out_chans=out_chans,
                         embed_dims=[24, 48, 96, 48, 24],
                         mlp_ratios=[2., 4., 4., 2., 2.],
                         depths=[4, 4, 4, 2, 2],
                         num_heads=[2, 4, 6, 1, 1],
                         attn_ratio=[0, 1 / 2, 1, 0, 0],
                         conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'])


def decloudformer_s(in_chans=3, out_chans=4, ):
    return DecloudFormer(in_chans=in_chans, out_chans=out_chans,
                         embed_dims=[24, 48, 96, 48, 24],
                         mlp_ratios=[2., 4., 4., 2., 2.],
                         depths=[8, 8, 8, 4, 4],
                         num_heads=[2, 4, 6, 1, 1],
                         attn_ratio=[1 / 4, 1 / 2, 3 / 4, 0, 0],
                         conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'])


def decloudformer_b(in_chans=3, out_chans=4, ):
    return DecloudFormer(in_chans=in_chans, out_chans=out_chans,
                         embed_dims=[24, 48, 96, 48, 24],
                         mlp_ratios=[2., 4., 4., 2., 2.],
                         depths=[16, 16, 16, 8, 8],
                         num_heads=[2, 4, 6, 1, 1],
                         attn_ratio=[1 / 4, 1 / 2, 3 / 4, 0, 0],
                         conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'])


def decloudformer_d(in_chans=3, out_chans=4, ):
    return DecloudFormer(in_chans=in_chans, out_chans=out_chans,
                         embed_dims=[24, 48, 96, 48, 24],
                         mlp_ratios=[2., 4., 4., 2., 2.],
                         depths=[32, 32, 32, 16, 16],
                         num_heads=[2, 4, 6, 1, 1],
                         attn_ratio=[1 / 4, 1 / 2, 3 / 4, 0, 0],
                         conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'])


def decloudformer_w(in_chans=3, out_chans=4, ):
    return DecloudFormer(in_chans=in_chans, out_chans=out_chans,
                         embed_dims=[48, 96, 192, 96, 48],
                         mlp_ratios=[2., 4., 4., 2., 2.],
                         depths=[16, 16, 16, 8, 8],
                         num_heads=[2, 4, 6, 1, 1],
                         attn_ratio=[1 / 4, 1 / 2, 3 / 4, 0, 0],
                         conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'])


def decloudformer_m(in_chans=3, out_chans=4, ):
    return DecloudFormer(in_chans=in_chans, out_chans=out_chans,
                         embed_dims=[24, 48, 96, 48, 24],
                         mlp_ratios=[2., 4., 4., 2., 2.],
                         depths=[12, 12, 12, 6, 6],
                         num_heads=[2, 4, 6, 1, 1],
                         attn_ratio=[1 / 4, 1 / 2, 3 / 4, 0, 0],
                         conv_type=['Conv', 'Conv', 'Conv', 'Conv', 'Conv'])


def decloudformer_l(in_chans=3, out_chans=4, ):
    return DecloudFormer(in_chans=in_chans, out_chans=out_chans,
                         embed_dims=[48, 96, 192, 96, 48],
                         mlp_ratios=[2., 4., 4., 2., 2.],
                         depths=[16, 16, 16, 12, 12],
                         num_heads=[2, 4, 6, 1, 1],
                         attn_ratio=[1 / 4, 1 / 2, 3 / 4, 0, 0],
                         conv_type=['Conv', 'Conv', 'Conv', 'Conv', 'Conv'],
                         window_size=7)


if __name__ == '__main__':
    model = decloudformer_l(in_chans=3, out_chans=4)
    print(model)

    x = torch.randn((2, 3, 224, 224))
    x = model(x)
    print(x.shape)