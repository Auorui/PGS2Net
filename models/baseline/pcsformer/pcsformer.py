import torch
import numpy as np
import torch.nn as nn
from timm.models.layers import trunc_normal_
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import DropPath

class Attention(nn.Module):
    def __init__(self, dim, resolution, stripeWidth, attention_mode, num_heads, currentDepth, attn_drop_rate, proj_drop_rata):
        super(Attention, self).__init__()
        self.resolution = resolution
        self.stripeWidth = stripeWidth
        self.attention_mode = attention_mode
        self.resolution = resolution
        self.num_heads = num_heads
        self.head_dim = dim//num_heads
        self.currentDepth = currentDepth

        self.scale = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.scale.data.fill_(self.head_dim ** -0.5)

        window_sizes = {
            -1: (resolution, resolution),
            0: (resolution, stripeWidth),
            1: (stripeWidth, resolution)
        }
        self.windowHeight, self.windowWidth = window_sizes.get(
            attention_mode, None)
        if self.windowHeight is None or self.windowWidth is None:
            print("ERROR MODE", attention_mode)

        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj_drop = nn.Dropout(proj_drop_rata)
        self.proj = nn.Linear(dim, dim)
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3,
                               stride=1, padding=1, groups=dim)

    def seq2CSwin(self, x):
        B, L, C = x.shape
        H = W = int(np.sqrt(L))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        x = x.view(B, C, H//self.windowHeight,
                   self.windowHeight, W//self.windowWidth, self.windowWidth).permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, self.windowHeight*self.windowWidth, C)
        x = x.reshape(-1, self.windowHeight*self.windowWidth,
                      self.num_heads, C//self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def CSwin2img(self, x, windowHeight, windowWidth, resolution):
        B = int(x.shape[0]/(resolution*resolution/windowHeight/windowWidth))
        x = x.view(B, resolution//windowHeight, resolution //
                   windowWidth, windowHeight, windowWidth, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(
            B, resolution, resolution, -1)
        return x

    def shiftedWindowVerticalPartition(self, x, reso, stripes):
        B, H, W, C = x.shape
        x = x.view(B, H//reso, reso, W//stripes, stripes, C)
        windows = x.permute(
            0, 1, 3, 2, 4, 5).contiguous().view(-1, reso*stripes, C)
        return windows

    def shiftedWindowHorizontalPartition(self, x, reso, stripes):
        B, H, W, C = x.shape
        x = x.view(B, H//stripes, stripes, W//reso, reso, C)
        windows = x.permute(
            0, 1, 3, 2, 4, 5).contiguous().view(-1, reso*stripes, C)
        return windows

    def verticalAttnMask(self, resolution, stripeWidth):
        shift_size = int(stripeWidth//2)
        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        img_mask = torch.zeros((1, resolution, resolution, 1), device=device)
        w_slices = (slice(0, -stripeWidth),
                    slice(-stripeWidth, -shift_size),
                    slice(-shift_size, None))
        cnt = 0
        for w in w_slices:
            img_mask[:, :, w, :] = cnt
            cnt += 1

        mask_windows = self.shiftedWindowVerticalPartition(
            img_mask, reso=resolution, stripes=stripeWidth)
        mask_windows = mask_windows.view(-1, resolution*stripeWidth)

        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(
            attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def horizontalMask(self, resolution, stripeWidth):
        shift_size = int(stripeWidth//2)

        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        img_mask = torch.zeros((1, resolution, resolution, 1), device=device)
        h_slices = (slice(0, -stripeWidth),
                    slice(-stripeWidth, -shift_size),
                    slice(-shift_size, None))
        cnt = 0
        for h in h_slices:
            img_mask[:, h, :, :] = cnt
            cnt += 1

        mask_windows = self.shiftedWindowHorizontalPartition(
            img_mask, reso=resolution, stripes=stripeWidth)
        mask_windows = mask_windows.view(-1, stripeWidth*resolution)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(
            attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def get_lepe(self, x, func):
        B, L, C = x.shape
        H = W = int(np.sqrt(L))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        H_sp, W_sp = self.windowHeight, self.windowWidth
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)

        lepe = func(x)
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads,
                            H_sp * W_sp).permute(0, 1, 3, 2).contiguous()
        x = x.reshape(-1, self.num_heads, C // self.num_heads,
                      self.windowHeight * self.windowWidth).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, qkv):
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Check size
        H = W = self.resolution
        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        # [B*(H//windowHeight)*(W//windowWidth), num_heads, windowHeight*windowWidth, C//num_heads]
        q, k = self.seq2CSwin(q), self.seq2CSwin(k)
        v, lepe = self.get_lepe(v, self.get_v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.currentDepth % 2 == 1:
            if self.attention_mode == 0:
                mask = self.verticalAttnMask(self.resolution, self.stripeWidth)
            elif self.attention_mode == 1:
                mask = self.horizontalMask(self.resolution, self.stripeWidth)
            mask = mask.repeat(B, 1, 1)
            mask = mask.unsqueeze(1)
            attn = attn + mask
            attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        else:
            attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v) + lepe
        x = x.transpose(1, 2).reshape(-1, self.windowHeight *
                                      self.windowWidth, C)
        x = self.proj_drop(self.proj(x))
        x = self.CSwin2img(x, windowHeight=self.windowHeight,
                           windowWidth=self.windowWidth, resolution=self.resolution).view(B, -1, C)

        return x


class SlidingCrossStripeTransformer(nn.Module):
    def __init__(self, dim, resolution, num_heads, stripeWidth, network_depth, mlp_ratio, qkv_bias, attn_drop_rate, proj_drop_rata, drop_path, norm_layer, currentDepth):
        super(SlidingCrossStripeTransformer, self).__init__()
        self.resolution = resolution
        self.currentDepth = currentDepth
        self.stripeWidth = stripeWidth

        self.no_cross_stripes = resolution == stripeWidth
        self.branch_num = 1 if self.no_cross_stripes else 2

        if self.no_cross_stripes:
            self.attns = nn.ModuleList([
                Attention(dim=dim, resolution=resolution, stripeWidth=stripeWidth, attention_mode=-1,
                          num_heads=num_heads, currentDepth=0, attn_drop_rate=attn_drop_rate, proj_drop_rata=proj_drop_rata)
                for i in range(self.branch_num)
            ])
        else:
            self.attns = nn.ModuleList([
                Attention(dim=dim//2, resolution=resolution, stripeWidth=stripeWidth, attention_mode=i,
                          num_heads=num_heads//2, currentDepth=currentDepth, attn_drop_rate=attn_drop_rate, proj_drop_rata=proj_drop_rata)
                for i in range(self.branch_num)
            ])

        mlp_hidden_dim = int(dim*mlp_ratio)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_dim=dim, network_depth=network_depth,
                       hidden_dim=mlp_hidden_dim, out_dim=dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)

    def shiftedWindowPartition(self, x):
        B, L, C = x.shape[1:]
        x = x.reshape(3, B, L//self.resolution, L//self.resolution, C)
        shift_size = self.stripeWidth//2
        x = torch.roll(
            x, shifts=(-shift_size, -shift_size), dims=(2, 3))
        x = x.reshape(3, B, -1, C)
        return x

    def reverseShiftedWindowPartition(self, x):
        B, L, C = x.shape
        x = x.reshape(B, L//self.resolution, L//self.resolution, C)
        shift_size = self.stripeWidth//2
        x = torch.roll(x, shifts=(shift_size, shift_size), dims=(1, 2))
        x = x.reshape(B, -1, C)
        return x

    def forward(self, x):
        H = W = self.resolution
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        qkv = self.qkv(x).reshape(
            B, -1, 3, C).permute(2, 0, 1, 3).contiguous()

        if self.currentDepth % 2 == 1 and self.no_cross_stripes == False:
            qkv = self.shiftedWindowPartition(qkv)

        qkv = self.norm1(qkv)

        attened_x = (
            torch.cat([self.attns[0](qkv[:, :, :, :C//2]),
                      self.attns[1](qkv[:, :, :, C//2:])], dim=2)
            if self.branch_num == 2
            else self.attns[0](qkv)
        )

        if self.currentDepth % 2 == 1 and self.no_cross_stripes == False:
            attened_x = self.reverseShiftedWindowPartition(attened_x)

        x = x + self.drop_path(attened_x)

        x = x + self.drop_path((self.mlp(self.norm2(x))))
        return x

class Mlp(nn.Module):
    def __init__(self, in_dim, network_depth, hidden_dim=None, out_dim=None):
        super(Mlp, self).__init__()
        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or in_dim
        self.network_depth = network_depth

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, out_dim),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            gain = (8 * self.network_depth) ** (-1/4)
            fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
            std = gain * np.sqrt(2.0 / float(fan_in + fan_out))
            trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)

class RelativePositionBias(nn.Module):
    def __init__(self, num_heads, h, w):
        super(RelativePositionBias, self).__init__()
        self.num_heads = num_heads
        self.h = int(h)
        self.w = int(w)

        self.relative_position_bias_table = nn.Parameter(
            torch.randn((2 * self.h - 1) * (2 * self.w - 1), self.num_heads) * 0.02)

        coords_h = torch.arange(self.h)
        coords_w = torch.arange(self.w)
        coords = torch.stack(torch.meshgrid(
            [coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)

        relative_coords = coords_flatten[:, :,
                                         None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.h - 1
        relative_coords[:, :, 1] += self.w - 1
        relative_coords[:, :, 0] *= 2 * self.h - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index",
                             relative_position_index)

    def forward(self, H, W):  # H and W is feature map size
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(
            -1)].view(self.h, self.w, self.h * self.w, -1)
        relative_position_bias_expand_h = torch.repeat_interleave(
            relative_position_bias, int(H // self.h), dim=0)
        relative_position_bias_expanded = torch.repeat_interleave(
            relative_position_bias_expand_h, int(W // self.w), dim=1)

        relative_position_bias_expanded = relative_position_bias_expanded.view(int(H * W), self.h * self.w,
                                                                               self.num_heads).permute(2, 0, 1).contiguous().unsqueeze(0)
        return relative_position_bias_expanded


class LocalProxyAttention(nn.Module):
    def __init__(self, dim, reso, num_heads, proxy_downscale, rel_pos, attn_drop_rate, proj_drop_rata):
        super(LocalProxyAttention, self).__init__()
        self.reso = reso
        self.proxy_downscale = proxy_downscale
        self.rel_pos = rel_pos
        self.num_heads = num_heads
        head_dim = dim//num_heads

        if rel_pos:
            self.relative_position_encoding = RelativePositionBias(
                num_heads=num_heads, h=reso//proxy_downscale, w=reso//proxy_downscale)

        self.scale = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.scale.data.fill_(head_dim ** -0.5)

        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_rata)

    def forward(self, qkv):
        _, B, L, C = qkv.shape
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q.reshape(B, L, self.num_heads, C //
                      self.num_heads).permute(0, 2, 1, 3).contiguous()
        k = k.reshape(B, L, self.num_heads, C //
                      self.num_heads).permute(0, 2, 1, 3).contiguous()
        v = v.reshape(B, L, self.num_heads, C //
                      self.num_heads).permute(0, 2, 1, 3).contiguous()
        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        if self.rel_pos:
            relative_position_bias = self.relative_position_encoding(
                self.reso/self.proxy_downscale, self.reso/self.proxy_downscale)
            attn = attn + relative_position_bias
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v)
        x = x.transpose(1, 2).reshape(-1, L, C)

        x = self.proj_drop(self.proj(x))

        return x


class LocalProxyTransformer(nn.Module):
    def __init__(self, dim, reso, proxy_downscale, num_heads, network_depth, mlp_ratio, rel_pos, qkv_bias, attn_drop_rate, proj_drop_rata, norm_layer):
        super(LocalProxyTransformer, self).__init__()

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.avgpool = nn.AvgPool2d(kernel_size=proxy_downscale)
        self.maxpool = nn.MaxPool2d(kernel_size=proxy_downscale)
        self.downsample = DownSample(
            in_channels=dim, out_channels=dim, patch_size=proxy_downscale, kernel_size=None)
        self.upsample = UpSample(
            in_channels=dim, out_channels=dim, patch_size=proxy_downscale, kernel_size=None)

        self.gate = nn.Conv2d(in_channels=dim*3, out_channels=3,
                              kernel_size=3, stride=1, padding=1, bias=True)

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim*mlp_ratio)
        self.mlp = Mlp(in_dim=dim, network_depth=network_depth,
                       hidden_dim=mlp_hidden_dim, out_dim=dim)

        self.attn = LocalProxyAttention(dim=dim, reso=reso, num_heads=num_heads, proxy_downscale=proxy_downscale,
                                        rel_pos=rel_pos, attn_drop_rate=attn_drop_rate, proj_drop_rata=proj_drop_rata)

    def seq2img(self, x):
        B, L, C = x.shape
        H = W = int(np.sqrt(L))
        x = x.permute(0, 2, 1).contiguous().reshape(B, C, H, W)
        return x

    def img2seq(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1).permute(0, 2, 1).contiguous()
        return x

    def forward(self, x):
        B, L, C = x.shape
        identity = x

        avg_proxy = self.avgpool(self.seq2img(x))
        max_proxy = self.maxpool(self.seq2img(x))
        conv_proxy = self.downsample(self.seq2img(x))
        gates = self.gate(torch.cat((avg_proxy, max_proxy, conv_proxy), dim=1))
        x = avg_proxy * gates[:, [0], :, :] + max_proxy * \
            gates[:, [1], :, :] + conv_proxy * gates[:, [2], :, :]
        x = self.img2seq(x)

        qkv = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3).contiguous()
        attened_x = self.attn(self.norm1(qkv))

        attened_x = self.upsample(self.seq2img(attened_x))
        x = identity + self.img2seq(attened_x)

        x = x + self.mlp(self.norm2(x))

        return x

class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class RefinementBlock(nn.Module):
    def __init__(self, dim, kernel_size):
        super(RefinementBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size,
                               stride=1, padding=(kernel_size//2), bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size,
                               stride=1, padding=(kernel_size//2), bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res

def bchw_2_blc(x):
    B, C, H, W = x.shape
    x = x.reshape(B, C, -1).permute(0, 2, 1)
    return x


def blc_2_bchw(x):
    B, L, C = x.shape
    H = W = int(np.sqrt(L))
    x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
    return x

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, kernel_size=None):
        super(DownSample, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        kernel_size = patch_size if kernel_size is None else kernel_size

        self.layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=patch_size, padding=(
            kernel_size-patch_size+1)//2, padding_mode='reflect')

    def forward(self, x):
        return self.layer(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, kernel_size=None):
        super(UpSample, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        kernel_size = 1 if kernel_size is None else kernel_size

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels*patch_size**2, kernel_size=kernel_size,
                      padding=kernel_size//2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        return self.layer(x)

class PatchEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=[7, 3], patch_size=2):
        super(PatchEmbed, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size_1, kernel_size_2 = kernel_size
        elif isinstance(kernel_size, list):
            kernel_size_1, kernel_size_2 = kernel_size[0], kernel_size[1]
        else:
            ValueError("kernel_size must be an integer or a list")

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size_1,
                      stride=patch_size, padding=(kernel_size_1-patch_size+1)//2, groups=in_channels, padding_mode='reflect'),
            nn.LeakyReLU(True),
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=kernel_size_2, stride=1, padding=1, groups=out_channels, padding_mode='reflect'),
            nn.LeakyReLU(True),
        )
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        return self.norm(bchw_2_blc(self.layer(x)))



class PCSformer(nn.Module):
    def __init__(self, in_channels=3, embed_dim=(48, 96, 96, 96, 48), depth = (4, 6, 8, 6, 4),
                 split_size=(1, 2, 8, 2, 1), proxy_downscale=(4, 2, 2), num_heads=(4, 8, 8, 4, 4),
                 mlp_ratio = 4, model_resolution = 256, qkv_bias = True, attn_drop_rate = 0.,
                 proj_drop_rata = 0., drop_path_rate = 0., num_refinement_blocks = 5,
                 refinement_block_dim = 48, norm_layer = nn.LayerNorm):
        super(PCSformer, self).__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depth))]  # stochastic depth decay rule

        self.patch_embed = PatchEmbed(
            in_channels, embed_dim[0], kernel_size=[7, 3], patch_size=2)

        self.stage1 = nn.ModuleList([
            SlidingCrossStripeTransformer(dim=embed_dim[0], resolution=model_resolution//2, num_heads=num_heads[0],
                                          stripeWidth=split_size[0], network_depth=sum(depth), mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, attn_drop_rate=attn_drop_rate, proj_drop_rata=proj_drop_rata,
                                          drop_path=dpr[i], norm_layer=nn.LayerNorm, currentDepth=i)
            for i in range(depth[0])
        ])
        self.LocalProxyTransformer1 = LocalProxyTransformer(
            dim=embed_dim[0], reso=model_resolution//2, proxy_downscale=proxy_downscale[0], num_heads=num_heads[0], network_depth=sum(depth),
            mlp_ratio=mlp_ratio, rel_pos=True, qkv_bias=qkv_bias, attn_drop_rate=attn_drop_rate, proj_drop_rata=proj_drop_rata, norm_layer=norm_layer)
        self.merge1 = DownSample(
            in_channels=embed_dim[0], out_channels=embed_dim[1], patch_size=2, kernel_size=5)
        self.stage1_3 = nn.Sequential(nn.Conv2d(in_channels=embed_dim[0], out_channels=embed_dim[0], kernel_size=4, stride=4, groups=embed_dim[0]),
                                      nn.Conv2d(in_channels=embed_dim[0], out_channels=embed_dim[0], kernel_size=1))
        self.stage1_4 = nn.Sequential(nn.Conv2d(in_channels=embed_dim[0], out_channels=embed_dim[0], kernel_size=2, stride=2, groups=embed_dim[0]),
                                      nn.Conv2d(in_channels=embed_dim[0], out_channels=embed_dim[0], kernel_size=1))

        self.stage2 = nn.ModuleList([
            SlidingCrossStripeTransformer(dim=embed_dim[1], resolution=model_resolution//4, num_heads=num_heads[1],
                                          stripeWidth=split_size[1], network_depth=sum(depth), mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, attn_drop_rate=attn_drop_rate, proj_drop_rata=proj_drop_rata,
                                          drop_path=dpr[np.sum(depth[:1])+i], norm_layer=nn.LayerNorm, currentDepth=i)
            for i in range(depth[1])
        ])
        self.LocalProxyTransformer2 = LocalProxyTransformer(
            dim=embed_dim[1], reso=model_resolution//4, proxy_downscale=proxy_downscale[1], num_heads=num_heads[1], network_depth=sum(depth),
            mlp_ratio=mlp_ratio, rel_pos=True, qkv_bias=qkv_bias, attn_drop_rate=attn_drop_rate, proj_drop_rata=proj_drop_rata, norm_layer=norm_layer)
        self.merge2 = DownSample(
            in_channels=embed_dim[1], out_channels=embed_dim[2], patch_size=2, kernel_size=3)
        self.stage2_5 = nn.Sequential(UpSample(in_channels=embed_dim[1], out_channels=embed_dim[1], patch_size=2, kernel_size=None),
                                      nn.Conv2d(embed_dim[1], embed_dim[0]//2, kernel_size=1))

        self.stage3 = nn.ModuleList([
            SlidingCrossStripeTransformer(dim=embed_dim[2], resolution=model_resolution//8, num_heads=num_heads[2],
                                          stripeWidth=split_size[2], network_depth=sum(depth), mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, attn_drop_rate=attn_drop_rate, proj_drop_rata=proj_drop_rata,
                                          drop_path=dpr[np.sum(depth[:2])+i], norm_layer=nn.LayerNorm, currentDepth=i)
            for i in range(depth[2])
        ])
        self.LocalProxyTransformer3 = LocalProxyTransformer(
            dim=embed_dim[2], reso=model_resolution//8, proxy_downscale=proxy_downscale[2], num_heads=num_heads[2], network_depth=sum(depth),
            mlp_ratio=mlp_ratio, rel_pos=True, qkv_bias=qkv_bias, attn_drop_rate=attn_drop_rate, proj_drop_rata=proj_drop_rata, norm_layer=norm_layer)
        self.upMerge1 = UpSample(
            in_channels=embed_dim[2], out_channels=embed_dim[3], patch_size=2, kernel_size=None)
        self.stage3_5 = nn.Sequential(UpSample(in_channels=embed_dim[2], out_channels=embed_dim[2], patch_size=4, kernel_size=None),
                                      nn.Conv2d(embed_dim[2], embed_dim[0]//2, kernel_size=1))

        self.stage4 = nn.ModuleList([
            SlidingCrossStripeTransformer(dim=embed_dim[3], resolution=model_resolution//4, num_heads=num_heads[3],
                                          stripeWidth=split_size[3], network_depth=sum(depth), mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, attn_drop_rate=attn_drop_rate, proj_drop_rata=proj_drop_rata,
                                          drop_path=dpr[np.sum(depth[:3])+i], norm_layer=nn.LayerNorm, currentDepth=i)
            for i in range(depth[3])
        ])
        self.upMerge2 = UpSample(
            in_channels=embed_dim[3], out_channels=embed_dim[4], patch_size=2, kernel_size=None)

        self.stage5 = nn.ModuleList([
            SlidingCrossStripeTransformer(dim=embed_dim[4], resolution=model_resolution//2, num_heads=num_heads[4],
                                          stripeWidth=split_size[4], network_depth=sum(depth), mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, attn_drop_rate=attn_drop_rate, proj_drop_rata=proj_drop_rata,
                                          drop_path=dpr[np.sum(depth[:4])+i], norm_layer=nn.LayerNorm, currentDepth=i)
            for i in range(depth[4])
        ])
        self.lastUpMetge = UpSample(
            in_channels=embed_dim[4], out_channels=embed_dim[4], patch_size=2, kernel_size=3)

        self.conv = nn.Conv2d(
            embed_dim[4], 4, kernel_size=3, stride=1, padding=1)

        self.proj3 = nn.Conv2d(
            embed_dim[0]+embed_dim[2], embed_dim[2], kernel_size=1)
        self.proj4 = nn.Conv2d(
            embed_dim[0]+embed_dim[1]+embed_dim[3], embed_dim[3], kernel_size=1)
        self.proj5 = nn.Conv2d(
            2*embed_dim[0]+embed_dim[4], embed_dim[4], kernel_size=1)

        self.refineProj1 = nn.Conv2d(
            3, refinement_block_dim, kernel_size=3, padding=1)
        self.refine_blocks = nn.ModuleList([
            RefinementBlock(dim=refinement_block_dim, kernel_size=3)
            for _ in range(num_refinement_blocks)
        ])
        self.refineProj2 = nn.Conv2d(
            refinement_block_dim, 3, kernel_size=3, padding=1)

    def forward_features(self, x):
        x = self.patch_embed(x)

        x1, x2 = x, x
        for blk in self.stage1:
            x1 = blk(x1)
        x2 = self.LocalProxyTransformer1(x2)
        x = blc_2_bchw(x1 + x2)
        skip1_3, skip1_4, skip1_5 = self.stage1_3(x), self.stage1_4(x), x

        x = bchw_2_blc(self.merge1(x))
        x1, x2 = x, x
        for blk in self.stage2:
            x1 = blk(x1)
        x2 = self.LocalProxyTransformer2(x2)
        x = blc_2_bchw(x1 + x2)
        skip2_5, skip2_4 = self.stage2_5(x), x

        x = bchw_2_blc(self.proj3(torch.cat((self.merge2(x), skip1_3), dim=1)))
        x1, x2 = x, x
        for blk in self.stage3:
            x1 = blk(x1)
        x2 = self.LocalProxyTransformer3(x2)
        x = blc_2_bchw(x1 + x2)
        skip3_5 = self.stage3_5(x)

        x = bchw_2_blc(self.proj4(
            torch.cat((self.upMerge1(x), skip1_4, skip2_4), dim=1)))
        for blk in self.stage4:
            x = blk(x)
        x = blc_2_bchw(x)

        x = bchw_2_blc(self.proj5(
            torch.cat((self.upMerge2(x), skip1_5, skip2_5, skip3_5), dim=1)))
        for blk in self.stage5:
            x = blk(x)

        return self.conv(self.lastUpMetge(blc_2_bchw(x)))

    def RefineNetwork(self, x):
        short_cut = x
        x = self.refineProj1(x)
        for block in self.refine_blocks:
            x = block(x)
        return short_cut + self.refineProj2(x)

    def forward(self, x):
        H, W = x.shape[2:]
        feat = self.forward_features(x)
        K, B = torch.split(feat, (1, 3), dim=1)

        x = K * x - B + x
        coarseDehazedImage = x[:, :, :H, :W]
        refinedDehazedImage = self.RefineNetwork(coarseDehazedImage)
        # return coarseDehazedImage, refinedDehazedImage
        return refinedDehazedImage

if __name__ == "__main__":
    from pyzjr.nn import summary_2, model_complexity_info
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PCSformer()
    model = model.to(device)
    inputs = torch.randn(1, 3, 256, 256).to(device)
    output = model(inputs)
    print("output", output.shape)
    summary_2(model, input_size=(3, 256, 256))
    model_complexity_info(model, (3, 256, 256))