# Heterogeneous Prompting Cross-Domain Modulation Network for Remote Sensing Image Dehazing
# TGRS 2026
# https://github.com/huangtaoer915/HPCDM
from einops.layers.torch import Rearrange
import math
PI = math.pi
import torchvision.ops as ops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.concat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn

class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn

class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2


# Fuesion_融合
class CGAFusion(nn.Module):
    # 来自DEANet, 原作者代码仓库没有
    def __init__(self, dim, reduction=8):
        super(CGAFusion, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        initial = x + y
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result

class RFFLayer(nn.Module):
    def __init__(self, in_channels, out_dim):
        super(RFFLayer, self).__init__()
        self.out_dim = out_dim
        self.in_channels = in_channels

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True),
        )
        # 固定 RFF 权重（注册 buffer）
        self.register_buffer("W", torch.randn(out_dim, in_channels // 8) * 10)
        self.register_buffer("b", torch.rand(out_dim) * 2 * PI)

    def forward(self, x):
        B, C, H, W = x.shape
        x_proj = self.proj(x)                     # [B, D, H, W]
        x_proj = x_proj.view(B, -1, H * W)        # [B, D, N]
        x_proj = x_proj.permute(0, 2, 1)          # [B, N, D]

        rff = torch.cos(torch.matmul(x_proj, self.W.t()) + self.b)  # [B, N, out_dim]
        rff = rff * np.sqrt(2.0 / self.out_dim)
        rff = rff.permute(0, 2, 1).view(B, self.out_dim, H, W)       # [B, out_dim, H, W]
        return rff

class DFconvResBlock(nn.Module):
    def __init__(self, in_channel, DF):
        super(DFconvResBlock, self).__init__()
        self.offset1 = nn.Conv2d(in_channel, DF * DF * 2, kernel_size=3, padding=1)
        self.deform1 = ops.DeformConv2d(in_channel, in_channel, kernel_size=DF, padding=DF // 2)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        offset1 = self.offset1(x)
        y1 = self.deform1(x, offset1)
        out = self.conv_cat(y1)
        return out + x

class DFFusionBlock(nn.Module):
    def __init__(self, in_channel):
        super(DFFusionBlock, self).__init__()
        self.df3 = DFconvResBlock(in_channel, DF=3)
        self.df5 = DFconvResBlock(in_channel, DF=5)
        self.df7 = DFconvResBlock(in_channel, DF=7)

        # 可学习融合权重（初始化为相等）
        self.weight3 = nn.Parameter(torch.tensor(1.0))
        self.weight5 = nn.Parameter(torch.tensor(1.0))
        self.weight7 = nn.Parameter(torch.tensor(1.0))

        # 输出通道与输入一致，可选1x1卷积调整融合后维度
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out3 = self.df3(x)
        out5 = self.df5(x)
        out7 = self.df7(x)

        # 归一化权重
        weights = torch.softmax(torch.stack([self.weight3, self.weight5, self.weight7]), dim=0)

        # 加权融合
        out = weights[0] * out3 + weights[1] * out5 + weights[2] * out7

        # 融合卷积
        out = self.fusion_conv(out)
        return out

def std_pooling(x):
    mean = torch.mean(x, dim=1, keepdim=True)
    mean_sq = torch.mean(x ** 2, dim=1, keepdim=True)
    std = torch.sqrt((mean_sq - mean ** 2).clamp(min=1e-6))
    return std

class ExpertModulationBlock(nn.Module):
    def __init__(self, mode=("avg", "max")):
        super().__init__()
        assert len(mode) == 2 and all(m in {"avg", "max", "std"} for m in mode), \
            f"mode must be a tuple of 2 elements from {'avg', 'max', 'std'}"
        self.mode = mode

        self.pooling = {
            "avg": lambda x: torch.mean(x, dim=1, keepdim=True),
            "max": lambda x: torch.max(x, dim=1, keepdim=True)[0],
            "std": std_pooling
        }

        self.downsample = nn.ModuleDict({
            "avg": nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1),
            "max": nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1),
            "std": nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
        })

        self.m_conv = nn.Sequential(
            nn.Conv2d(1, 1, 3, padding=1), nn.LeakyReLU(), nn.Dropout(0.3),
            nn.Conv2d(1, 1, 3, padding=1), nn.LeakyReLU(), nn.Dropout(0.3),
            nn.Conv2d(1, 1, 3, padding=1), nn.Tanh()
        )

        self.b_conv = nn.Sequential(
            nn.Conv2d(1, 1, 3, padding=1), nn.LeakyReLU(), nn.Dropout(0.3),
            nn.Conv2d(1, 1, 3, padding=1), nn.LeakyReLU(), nn.Dropout(0.3),
            nn.Conv2d(1, 1, 3, padding=1)
        )

        self.conv = nn.Conv2d(2, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        f1_name, f2_name = self.mode
        f1 = self.pooling[f1_name](x)
        f2 = self.pooling[f2_name](x)

        f1_down = self.downsample[f1_name](f1)
        f2_down = self.downsample[f2_name](f2)

        w1 = self.m_conv(f1_down)
        b1 = self.b_conv(f1_down)
        w2 = self.m_conv(f2_down)
        b2 = self.b_conv(f2_down)

        modulated_2 = f2_down * w1 + b1
        modulated_1 = f1_down * w2 + b2

        fused = torch.cat([modulated_1, modulated_2], dim=1)
        out = self.conv(fused)
        mask = self.sigmoid(self.upsample(out))
        return F.relu(x * mask)

class GateNetwork(nn.Module):
    def __init__(self, input_size, num_experts=4, top_k=3):
        super(GateNetwork, self).__init__()
        self.gap = nn.AdaptiveMaxPool2d(1)
        self.gap2 = nn.AdaptiveAvgPool2d(1)
        self.input_size = input_size
        self.num_experts = num_experts
        self.top_k = top_k

        self.fc0 = nn.Linear(input_size, num_experts)
        self.fc1 = nn.Linear(input_size, num_experts)
        self.relu1 = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=1)
        init.zeros_(self.fc1.weight)
        self.sp = nn.Softplus()

    def forward(self, x):
        x = self.gap(x) + self.gap2(x)
        x = x.view(-1, self.input_size)
        inp = x

        x = self.fc1(x)
        x = self.relu1(x)
        noise = self.sp(self.fc0(inp))
        noise_mean = torch.mean(noise, dim=1, keepdim=True)
        std = torch.std(noise, dim=1, keepdim=True) + 1e-6
        noram_noise = (noise - noise_mean) / std

        topk_values, topk_indices = torch.topk(x + noram_noise, k=self.top_k, dim=1)
        mask = torch.zeros_like(x).scatter_(dim=1, index=topk_indices, value=1.0)
        x[~mask.bool()] = float('-inf')
        gating_coeffs = self.softmax(x)

        return gating_coeffs

class MoFE(nn.Module):
    def __init__(self, dim, flag):
        super(MoFE, self).__init__()
        self.dim = dim
        self.num_groups = 3
        self.num_experts_per_group = 4
        self.top_k_per_group = 3


        self.group_modes = [
            ("avg", "max"),
            ("std", "max"),
            ("std", "avg")
        ]


        self.gates = nn.ModuleList([
            GateNetwork(
                input_size=dim,
                num_experts=self.num_experts_per_group,
                top_k=self.top_k_per_group
            ) for _ in range(self.num_groups)
        ])


        self.expert_groups = nn.ModuleList()
        for mode in self.group_modes:
            group = nn.ModuleList()

            for _ in range(self.num_experts_per_group):
                if flag == 'C':
                    expert = nn.Sequential(
                        nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
                        ExpertModulationBlock(mode=mode),
                        nn.ReLU(),
                        nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
                    )
                else:
                    expert = nn.Sequential(
                        nn.Conv2d(dim, dim, 3, 1, padding=2, groups=dim, dilation=2),
                        ExpertModulationBlock(mode=mode),
                        nn.ReLU(),
                        nn.Conv2d(dim, dim, 3, 1, padding=2, groups=dim, dilation=2)
                    )
                group.append(expert)
            self.expert_groups.append(group)


        self.output_fusion = nn.Conv2d(dim * self.num_groups, dim, kernel_size=1)

    def forward(self, x):
        group_outputs = []

        for group_idx in range(self.num_groups):
            gate_coeffs = self.gates[group_idx](x)

            group_out = torch.zeros_like(x).to(x.device)

            for expert_idx in range(self.num_experts_per_group):
                if torch.all(gate_coeffs[:, expert_idx] == 0):
                    continue

                active_samples = torch.where(gate_coeffs[:, expert_idx] > 0)[0]
                if len(active_samples) == 0:
                    continue

                expert = self.expert_groups[group_idx][expert_idx]
                expert_out = expert(x[active_samples])

                coeff = gate_coeffs[active_samples, expert_idx].view(-1, 1, 1, 1)
                group_out[active_samples] += expert_out * coeff

            group_outputs.append(group_out)

        fused = torch.cat(group_outputs, dim=1)
        final_out = self.output_fusion(fused)

        return final_out

class DynamicFusionHazeMappingNetwork(nn.Module):

    def __init__(self, in_channels=3, dim=64, flag='C'):
        super(DynamicFusionHazeMappingNetwork, self).__init__()
        self.initial_conv = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1)

        self.rff = RFFLayer(in_channels=dim, out_dim=dim)
        self.dffusion = DFFusionBlock(in_channel=dim)
        self.mofe = MoFE(dim=dim, flag=flag)

        self.residual_conv = nn.Conv2d(in_channels, dim, kernel_size=1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):

        input_shape = x.shape


        residual = self.residual_conv(x)


        x = self.initial_conv(x)


        x = self.rff(x)  # [B, dim, H, W]
        x = self.dffusion(x)  # [B, dim, H, W]
        x = self.mofe(x)  # [B, dim, H, W]


        x = x + residual  # [B, dim, H, W]


        haze_distribution = self.output_layer(x)  # [B, in_channels, H, W]


        assert haze_distribution.shape == input_shape, \
            f"输出形状 {haze_distribution.shape} 与输入形状 {input_shape} 不一致"

        return haze_distribution

###################################mokuai2#######################################
class GlobalFourierPromptModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Amplitude branch
        self.amp_conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.amp_relu  = nn.ReLU(inplace=True)
        self.amp_conv2 = nn.Conv2d(channels, channels, kernel_size=1)

        # Channel attention (recommended for amplitude branch or weight generation)
        self.ca = ChannelAttention(channels)
        self.pa = SpatialAttention()
        # Prompt modulation: depthwise affine transform
        self.W = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels)
        self.b = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels)

        self.conv=nn.Conv2d(channels*3 , channels, kernel_size=1, stride=1, padding=0)
    def forward(self, haze, prompt):

        haze_f   = torch.fft.fft2(haze)      # complex
        prompt_f = torch.fft.fft2(prompt)

        haze_amp = torch.abs(haze_f)         # amplitude of haze
        haze_pha = torch.angle(haze_f)       # phase of haze (radians)

        prompt_amp = torch.abs(prompt_f)     # amplitude of prompt
        prompt_pha = torch.angle(prompt_f)     # amplitude of prompt


        amp_feat = self.amp_conv2(self.amp_relu(self.amp_conv1(haze_amp)))
        amp_gate = self.ca(amp_feat * prompt_amp)
        out_amp  = amp_feat * torch.sigmoid(amp_gate)+haze_amp


        pha_feat=self.amp_conv2(self.amp_relu(self.amp_conv1(haze_pha)))
        phi_res  = pha_feat-prompt_pha
        pha_gate = self.ca(phi_res)
        out_pha  = torch.sigmoid(pha_gate) * haze_pha+haze_pha

        real_hazy = out_amp * torch.cos(out_pha)
        imag_hazy = out_amp * torch.sin(out_pha)
        fre_out = torch.complex(real_hazy, imag_hazy)
        fre_out=torch.fft.ifft2(fre_out, norm='backward').real

        prompt_w=self.W(prompt)
        prompt_b=self.b(prompt)
        spa_res = haze - fre_out
        feat_pa = self.pa(spa_res)
        feature1=haze
        feature2=torch.sigmoid(feat_pa)*haze
        feature3=haze*prompt_w+prompt_b+feature2

        feature=torch.cat([feature1,feature2,feature3],dim=1)
        feature=self.conv(feature)

        result=haze+feature

        return result

###################################Net#######################################
class HPCDMNet(nn.Module):
    def __init__(self, base_dim=32):
        super(HPCDMNet, self).__init__()
        self.prompt_feature = DynamicFusionHazeMappingNetwork()
        self.GFP1 = GlobalFourierPromptModule(channels=base_dim)
        self.GFP2 = GlobalFourierPromptModule(channels=2*base_dim)
        self.GFP3 = GlobalFourierPromptModule(channels=4*base_dim)
        self.GFP4 = GlobalFourierPromptModule(channels=2*base_dim)
        self.GFP5 = GlobalFourierPromptModule(channels=base_dim)
        self.prompt_conv1 = nn.Conv2d(3, base_dim, kernel_size=3, padding=1)
        self.prompt_conv2 = nn.Sequential(
            nn.Conv2d(3, 2 * base_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True)
        )
        self.prompt_conv3 = nn.Sequential(
            nn.Conv2d(3, 2 * base_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(2 * base_dim, 4 * base_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True)
        )
        # down-sample
        self.down1 = nn.Sequential(nn.Conv2d(3, base_dim, kernel_size=3, stride = 1, padding=1))
        self.down2 = nn.Sequential(nn.Conv2d(base_dim, base_dim*2, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))
        self.down3 = nn.Sequential(nn.Conv2d(base_dim*2, base_dim*4, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))
        # level1

        # level2
        self.fe_level_2 = nn.Conv2d(in_channels=base_dim * 2, out_channels=base_dim * 2, kernel_size=3, stride=1, padding=1)

        # level3
        self.fe_level_3 = nn.Conv2d(in_channels=base_dim * 4, out_channels=base_dim * 4, kernel_size=3, stride=1, padding=1)

        # up-sample
        self.up1 = nn.Sequential(nn.ConvTranspose2d(base_dim*4, base_dim*2, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(base_dim*2, base_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up3 = nn.Sequential(nn.Conv2d(base_dim, 3, kernel_size=3, stride=1, padding=1))
        # feature fusion
        self.mix1 = CGAFusion(base_dim * 4, reduction=8)
        self.mix2 = CGAFusion(base_dim * 2, reduction=4)

    def forward(self, x):
        prompt = self.prompt_feature(x)
        prompt1 = self.prompt_conv1(prompt)  # (B, base_dim, 256, 256)
        prompt2 = self.prompt_conv2(prompt)  # (B, 2*base_dim, 128, 128)
        prompt3 = self.prompt_conv3(prompt)  # (B, 4*base_dim, 64, 64)

        x_down1 = self.down1(x)
        x_down1 = self.down_level1_block1(x_down1)
        x_down1 = self.down_level1_block2(x_down1)
        x_down1 = self.down_level1_block3(x_down1)
        x_down1 = self.down_level1_block4(x_down1)
        gfp1=self.GFP1(x_down1,prompt1)
        x_down1 = x_down1 + gfp1

        x_down2 = self.down2(x_down1)
        x_down2_init = self.fe_level_2(x_down2)
        x_down2_init = self.down_level2_block1(x_down2_init)
        x_down2_init = self.down_level2_block2(x_down2_init)
        x_down2_init = self.down_level2_block3(x_down2_init)
        x_down2_init = self.down_level2_block4(x_down2_init)
        gfp2 = self.GFP2(x_down2_init, prompt2)
        x_down2_init = x_down2_init + gfp2

        x_down3 = self.down3(x_down2_init)
        x_down3_init = self.fe_level_3(x_down3)
        x1 = self.level3_block1(x_down3_init)
        x2 = self.level3_block2(x1)
        x3 = self.level3_block3(x2)
        x4 = self.level3_block4(x3)
        x5 = self.level3_block5(x4)
        x6 = self.level3_block6(x5)
        x7 = self.level3_block7(x6)
        x8 = self.level3_block8(x7)
        gfp3 = self.GFP3(x8, prompt3)
        x8 = x8 + gfp3
        x_level3_mix = self.mix1(x_down3, x8)

        x_up1 = self.up1(x_level3_mix)
        x_up1 = self.up_level2_block1(x_up1)
        x_up1 = self.up_level2_block2(x_up1)
        x_up1 = self.up_level2_block3(x_up1)
        x_up1 = self.up_level2_block4(x_up1)
        gfp4 = self.GFP4(x_up1, prompt2)
        x_up1 = x_up1 + gfp4
        x_level2_mix = self.mix2(x_down2, x_up1)
        x_up2 = self.up2(x_level2_mix)
        x_up2 = self.up_level1_block1(x_up2)
        x_up2 = self.up_level1_block2(x_up2)
        x_up2 = self.up_level1_block3(x_up2)
        x_up2 = self.up_level1_block4(x_up2)
        gfp5 = self.GFP5(x_up2, prompt1)
        x_up2 = x_up2 + gfp5
        out = self.up3(x_up2)

        # return out,prompt
        return out