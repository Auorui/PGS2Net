import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.layers import to_2tuple, trunc_normal_
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_sizes=[3, 5, 7]):
        super(SpatialAttention, self).__init__()
        assert all(k in (3, 5, 7) for k in kernel_sizes), 'kernel sizes must be 3, 5, or 7'
        self.convs = nn.ModuleList([
            nn.Conv2d(2, 1, kernel_size=k, padding=k // 2, bias=False) for k in kernel_sizes
        ])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.contiguous()
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        
        conv_outs = [conv(x) for conv in self.convs]
        x = torch.sum(torch.stack(conv_outs), dim=0)
        
        return self.sigmoid(x)
class TransformerBlock(nn.Module):
    def __init__(self, network_depth, dim, num_heads,
                 window_size=8):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.attn = Attention(network_depth, dim, num_heads=num_heads, shift_size = 0, window_size=window_size)

    def forward(self, x, prompt):
        B, C, H, W = x.shape
        identity = x
        x = x.view(B, C, H * W)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = x.view(B, C, H, W)
        x = self.attn(x, prompt)
        x = identity + x

        return x

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

    coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
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

    def forward(self, qkv):
        B_, N, _ = qkv.shape

        qkv = qkv.reshape(B_, N, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.meta(self.relative_positions)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, self.dim)
        return x
    
class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=2, out_chans=3, in_chans=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = in_chans

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, out_chans*patch_size**2, kernel_size=kernel_size,
                      padding=kernel_size//2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
        return x
class Attention(nn.Module):
    def __init__(self, network_depth, dim, num_heads, window_size, shift_size):
        super().__init__()
        self.dim = dim
        self.head_dim = int(dim // num_heads)
        self.num_heads = num_heads

        self.window_size = window_size
        self.shift_size = shift_size

        self.network_depth = network_depth

        self.V = nn.Conv2d(dim, dim, 1)
        self.Q = nn.Conv2d(dim, dim, 1)
        self.K = nn.Conv2d(dim, dim, 1)
        self.ca = ChannelAttention(dim)
        self.sa = SpatialAttention(kernel_sizes=[3,5,7])
        self.attn = WindowAttention(dim, window_size, num_heads)
            
        self.proj = nn.Conv2d(dim, dim, 1)
        self.conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode='reflect')
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            w_shape = m.weight.shape
            
            if w_shape[0] == self.dim * 2:	# QK
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)		
            else:
                gain = (8 * self.network_depth) ** (-1/4)
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
            x = F.pad(x, (self.shift_size, (self.window_size-self.shift_size+mod_pad_w) % self.window_size,
                          self.shift_size, (self.window_size-self.shift_size+mod_pad_h) % self.window_size), mode='reflect')
        else:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, X, prompt):
        B, C, H, W = X.shape
        V = self.V(X)
        w_ca = self.ca(prompt)
        w_sa = self.sa(prompt)
        X_p = w_ca * X
        X_p = w_sa * X_p

        Q = self.Q(X)+ X_p
        # Q = self.Q(X)
        K = self.K(X)

        QKV = torch.cat([Q, K, V], dim=1)

        # shift
        shifted_QKV = self.check_size(QKV, self.shift_size > 0)
        Ht, Wt = shifted_QKV.shape[2:]

        # partition windows
        shifted_QKV = shifted_QKV.permute(0, 2, 3, 1)
        qkv = window_partition(shifted_QKV, self.window_size)  # nW*B, window_size**2, C

        attn_windows = self.attn(qkv)

        # merge windows
        shifted_out = window_reverse(attn_windows, self.window_size, Ht, Wt)  # B H' W' C

        # reverse cyclic shift
        out = shifted_out[:, self.shift_size:(self.shift_size+H), self.shift_size:(self.shift_size+W), :]
        attn_out = out.permute(0, 3, 1, 2)

        conv_out = self.conv(V)
        out = self.proj(conv_out + attn_out)

        return out
