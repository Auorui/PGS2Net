# Cross-Frequency Attention and Color Contrast Constraint for Remote Sensing Dehazing
# TIP 2025
# https://github.com/fyxnl/C4RSD/tree/master
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.baseline.c4rsd.transformer import TransformerBlock
from models.baseline.c4rsd.mamba import Vim
from models.baseline.c4rsd.haar_wavelet import HaarWavelet
from models.baseline.c4rsd.res2net_v1b import build_res2net101_encoder


class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation != 'no':
            return self.act(out)
        else:
            return out


class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class Decoder_MDCBlock1(torch.nn.Module):
    def __init__(self, num_filter, num_ft, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm=None,
                 mode='iter1'):
        super(Decoder_MDCBlock1, self).__init__()
        self.mode = mode
        self.num_ft = num_ft - 1
        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for i in range(self.num_ft):
            self.down_convs.append(
                ConvBlock(num_filter * (2 ** i), num_filter * (2 ** (i + 1)), kernel_size, stride, padding, bias,
                          activation, norm=None)
            )
            self.up_convs.append(
                DeconvBlock(num_filter * (2 ** (i + 1)), num_filter * (2 ** i), kernel_size, stride, padding, bias,
                            activation, norm=None)
            )

    def forward(self, ft_h, ft_l_list):
        if self.mode == 'iter1' or self.mode == 'conv':
            ft_h_list = []
            for i in range(len(ft_l_list)):
                ft_h_list.append(ft_h)
                ft_h = self.down_convs[self.num_ft - len(ft_l_list) + i](ft_h)

            ft_fusion = ft_h
            for i in range(len(ft_l_list)):
                ft_fusion = self.up_convs[self.num_ft - i - 1](ft_fusion - ft_l_list[i]) + ft_h_list[
                    len(ft_l_list) - i - 1]

        if self.mode == 'iter2':
            ft_fusion = ft_h
            for i in range(len(ft_l_list)):
                ft = ft_fusion
                for j in range(self.num_ft - i):
                    ft = self.down_convs[j](ft)

                ft = ft - ft_l_list[i]
                for j in range(self.num_ft - i):
                    ft = self.up_convs[self.num_ft - i - j - 1](ft)
                ft_fusion = ft_fusion + ft

        if self.mode == 'iter3':
            ft_fusion = ft_h
            for i in range(len(ft_l_list)):
                ft = ft_fusion
                for j in range(i + 1):
                    ft = self.down_convs[j](ft)
                ft = ft - ft_l_list[len(ft_l_list) - i - 1]
                for j in range(i + 1):
                    # print(j)
                    ft = self.up_convs[i + 1 - j - 1](ft)
                ft_fusion = ft_fusion + ft

        if self.mode == 'iter4':
            ft_fusion = ft_h
            for i in range(len(ft_l_list)):
                ft = ft_h
                for j in range(self.num_ft - i):
                    ft = self.down_convs[j](ft)
                ft = ft - ft_l_list[i]
                for j in range(self.num_ft - i):
                    ft = self.up_convs[self.num_ft - i - j - 1](ft)
                ft_fusion = ft_fusion + ft

        return ft_fusion


class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


# Residual dense block (RDB) architecture
class RDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate, scale=1.0):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        self.scale = scale
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out) * self.scale
        out = out + x
        return out


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ConvLayer1(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(UpsampleConvLayer, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        h = (x.shape[2] - 1) * self.stride + self.kernel_size
        w = (x.shape[3] - 1) * self.stride + self.kernel_size
        x = F.interpolate(x, size=(h, w), mode="nearest-exact")
        out = self.conv2d(x)
        return out


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.relu = nn.PReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out


class SE(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        """
        Squeeze-and-Excitation Block
        :param channels: Number of input channels
        :param reduction: Reduction ratio for the hidden layer
        """
        super(SE, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.fc = nn.Sequential(
            nn.Linear(out_channels, out_channels // reduction, bias=False),  # Reduce channels
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // reduction, out_channels, bias=False),  # Restore channels
            nn.Sigmoid()  # Generate channel-wise weights
        )

    def forward(self, x):
        x = self.conv(x)
        batch_size, channels, _, _ = x.size()
        # Squeeze: Global Average Pooling
        y = self.global_avg_pool(x).view(batch_size, channels)

        # Excitation: Fully Connected Layers
        y = self.fc(y).view(batch_size, channels, 1, 1)

        # Scale: Channel-wise multiplication
        return x * y


class decoder(nn.Module):
    def __init__(self, res_blocks=18):
        super(decoder, self).__init__()
        self.layer = BasicBlock(dim=256, num_blocks=10)
        self.dehaze = nn.Sequential()
        for i in range(0, res_blocks):
            self.dehaze.add_module('res%d' % i, ResidualBlock(256))

        self.convd16x = UpsampleConvLayer(256, 128, kernel_size=3, stride=2)
        self.dense_4 = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )
        self.conv_4 = RDB(64, 4, 64)
        self.fusion_4 = Decoder_MDCBlock1(64, 2, mode='iter2')

        self.convd8x = UpsampleConvLayer(128, 64, kernel_size=3, stride=2)
        self.dense_3 = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64)
        )
        self.conv_3 = RDB(32, 4, 32)
        self.fusion_3 = Decoder_MDCBlock1(32, 3, mode='iter2')

        self.convd4x = UpsampleConvLayer(64, 32, kernel_size=3, stride=2)
        self.dense_2 = nn.Sequential(
            ResidualBlock(32),
            ResidualBlock(32),
            ResidualBlock(32)
        )
        self.conv_2 = RDB(16, 4, 16)
        self.fusion_2 = Decoder_MDCBlock1(16, 4, mode='iter2')

        self.convd2x = UpsampleConvLayer(32, 16, kernel_size=3, stride=2)
        self.dense_1 = nn.Sequential(
            ResidualBlock(16),
            ResidualBlock(16),
            ResidualBlock(16)
        )
        self.conv_1 = RDB(8, 4, 8)
        self.fusion_1 = Decoder_MDCBlock1(8, 5, mode='iter2')

        self.conv_output = ConvLayer(16, 3, kernel_size=3, stride=1)

        self.C0 = SE(1024, 256)
        self.C1 = SE(512, 128)
        self.C2 = SE(256, 64)
        self.C3 = SE(64, 32)

    def forward(self, x, x_e, pH, pV, pD):
        ini = x

        res16x = self.C0(x_e[0])
        res8x = self.C1(x_e[1])
        res4x = self.C2(x_e[2])
        res2x = self.C3(x_e[3])

        res16x = self.layer(res16x, pH, pV, pD)

        res_dehaze = res16x
        in_ft = res16x * 2
        res16x = self.dehaze(in_ft) + in_ft - res_dehaze
        res16x_1, res16x_2 = res16x.split([(res16x.size()[1] // 2), (res16x.size()[1] // 2)], dim=1)
        feature_mem_up = [res16x_1]

        res16x = self.convd16x(res16x)

        res16x = F.interpolate(res16x, res8x.size()[2:], mode='bilinear')
        res8x = torch.add(res16x, res8x)
        res8x = self.dense_4(res8x) + res8x - res16x
        res8x_1, res8x_2 = res8x.split([(res8x.size()[1] // 2), (res8x.size()[1] // 2)], dim=1)
        res8x_1 = self.fusion_4(res8x_1, feature_mem_up)
        res8x_2 = self.conv_4(res8x_2)
        feature_mem_up.append(res8x_1)
        res8x = torch.cat((res8x_1, res8x_2), dim=1)

        res8x = self.convd8x(res8x)

        res8x = F.interpolate(res8x, res4x.size()[2:], mode='bilinear')
        res4x = torch.add(res8x, res4x)
        res4x = self.dense_3(res4x) + res4x - res8x
        res4x_1, res4x_2 = res4x.split([(res4x.size()[1] // 2), (res4x.size()[1] // 2)], dim=1)
        res4x_1 = self.fusion_3(res4x_1, feature_mem_up)
        res4x_2 = self.conv_3(res4x_2)
        feature_mem_up.append(res4x_1)
        res4x = torch.cat((res4x_1, res4x_2), dim=1)

        res4x = self.convd4x(res4x)

        res4x = F.interpolate(res4x, res2x.size()[2:], mode='bilinear')
        res2x = torch.add(res4x, res2x)

        res2x = self.dense_2(res2x) + res2x - res4x
        res2x_1, res2x_2 = res2x.split([(res2x.size()[1] // 2), (res2x.size()[1] // 2)], dim=1)
        res2x_1 = self.fusion_2(res2x_1, feature_mem_up)
        res2x_2 = self.conv_2(res2x_2)

        feature_mem_up.append(res2x_1)
        res2x = torch.cat((res2x_1, res2x_2), dim=1)
        res2x = self.convd2x(res2x)

        res2x = F.interpolate(res2x, ini.size()[2:], mode='bilinear')
        x = torch.add(res2x, res2x)
        x = self.dense_1(x) + x - res2x
        x_1, x_2 = x.split([(x.size()[1] // 2), (x.size()[1] // 2)], dim=1)
        x_1 = self.fusion_1(x_1, feature_mem_up)

        x_2 = self.conv_1(x_2)
        x = torch.cat((x_1, x_2), dim=1)

        x = self.conv_output(x)

        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, in_chans=96, out_chans=24, patch_size=2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels=in_chans, out_channels=out_chans * patch_size ** 2, kernel_size=1, stride=1,
                      padding=0),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, out_chans=96, patch_size=4, kernel_size=None):
        super().__init__()
        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_chans, out_chans, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')

    def forward(self, x):
        x = self.proj(x)
        return x


class mamba_Branches(nn.Module):
    def __init__(self, scan_mode='H'):
        super().__init__()

        self.up = PatchUnEmbed(in_chans=256, out_chans=16, patch_size=8)
        self.down = PatchEmbed(in_chans=16, out_chans=256, patch_size=8)
        self.mamba = Vim(dim=256, dt_rank=32, d_state=32, dropout=0.1, depth=1, scan_mode=scan_mode)
        self.conv_output = ConvLayer(16, 3, kernel_size=3, stride=1)
        self.conv_input = ConvLayer(3, 16, kernel_size=3, stride=1)

    def forward(self, c):
        ini = c
        c = self.conv_input(c)
        c = self.down(c)
        prompt = self.mamba(c)
        c = self.up(prompt)
        c = self.conv_output(c) + ini

        return prompt, c

class mamba_Branches2(nn.Module):
    def __init__(self, scan_mode='H'):
        super().__init__()

        # 输入映射
        self.conv_input = ConvLayer(3, 16, kernel_size=3, stride=1)

        # 编码器：PatchEmbed 下采样
        self.down1 = PatchEmbed(in_chans=16, out_chans=64, patch_size=2)    # H/2
        self.down2 = PatchEmbed(in_chans=64, out_chans=128, patch_size=2)   # H/4
        self.down3 = PatchEmbed(in_chans=128, out_chans=256, patch_size=2)  # H/8

        # Mamba中间块
        self.mamba = Vim(dim=256, dt_rank=32, d_state=64, dropout=0.1, depth=2, scan_mode=scan_mode)

        # 解码器：PatchUnEmbed 上采样
        self.up3 = PatchUnEmbed(in_chans=256, out_chans=128, patch_size=2)  # H/4
        self.up2 = PatchUnEmbed(in_chans=256, out_chans=64, patch_size=2)   # H/2
        self.up1 = PatchUnEmbed(in_chans=128, out_chans=16, patch_size=2)   # H

        # 输出映射
        self.conv_output = ConvLayer(16, 3, kernel_size=3, stride=1)

    def forward(self, x):
        x_in = x  # 原图保留用于残差
        x = self.conv_input(x)  # [B, 16, H, W]

        # 编码路径
        x1 = self.down1(x)   # [B, 64, H/2, W/2]
        x2 = self.down2(x1)  # [B, 128, H/4, W/4]
        x3 = self.down3(x2)  # [B, 256, H/8, W/8]

        x3_mamba = self.mamba(x3)

        # 解码路径 + 拼接跳跃连接
        d3 = self.up3(x3)               # [B, 128, H/4, W/4]
        d3 = torch.cat([d3, x2], dim=1) # [B, 256, H/4, W/4]

        d2 = self.up2(d3)               # [B, 64, H/2, W/2]
        d2 = torch.cat([d2, x1], dim=1) # [B, 128, H/2, W/2]

        d1 = self.up1(d2)               # [B, 16, H, W]
        out = self.conv_output(d1) + x_in  # 残差连接

        return x3_mamba, out

class mamba_Branches3(nn.Module):
    def __init__(self, scan_mode='H'):
        super().__init__()

        self.conv_input = ConvLayer1(3, 16)

        # 编码路径 + 下采样
        self.down1 = PatchEmbed(in_chans=16, out_chans=64, patch_size=2)
        self.enc_conv1 = ConvLayer1(64, 64)

        self.down2 = PatchEmbed(in_chans=64, out_chans=128, patch_size=2)
        self.enc_conv2 = ConvLayer1(128, 128)

        self.down3 = PatchEmbed(in_chans=128, out_chans=256, patch_size=2)
        self.enc_conv3 = ConvLayer1(256, 256)

        # Mamba 模块
        self.mamba = Vim(dim=256, dt_rank=32, d_state=64, dropout=0.1, depth=1, scan_mode=scan_mode)

        # 解码路径 + 上采样
        self.up3 = PatchUnEmbed(in_chans=256, out_chans=128, patch_size=2)
        self.dec_conv3 = ConvLayer1(256, 128)

        self.up2 = PatchUnEmbed(in_chans=128, out_chans=64, patch_size=2)
        self.dec_conv2 = ConvLayer1(128, 64)

        self.up1 = PatchUnEmbed(in_chans=64, out_chans=16, patch_size=2)
        self.dec_conv1 = ConvLayer1(32, 16)

        self.conv_output = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x_in = x
        x = self.conv_input(x)

        x1 = self.down1(x)
        x1 = self.enc_conv1(x1)

        x2 = self.down2(x1)
        x2 = self.enc_conv2(x2)

        x3 = self.down3(x2)
        x3 = self.enc_conv3(x3)

        x3_mamba = self.mamba(x3)

        # 解码路径
        d3 = self.up3(x3)
        d3 = torch.cat([d3, x2], dim=1)
        d3 = self.dec_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, x1], dim=1)
        d2 = self.dec_conv2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, x], dim=1)
        d1 = self.dec_conv1(d1)

        out = self.conv_output(d1) + x_in

        return x3_mamba, out


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


class TambaBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        layers = [TransformerBlock(dim=dim, network_depth=30, num_heads=2, window_size=8),
                  TransformerBlock(dim=dim, network_depth=30, num_heads=2, window_size=8),
                  TransformerBlock(dim=dim, network_depth=30, num_heads=2, window_size=8), ]
        self.layers = nn.Sequential(*layers)
        self.ca = ChannelAttention(dim * 3)
        self.sa = SpatialAttention(kernel_sizes=[3, 5, 7])

    def forward(self, x, mamba16H, mamba16V, mamba16D):
        res1 = self.layers[0](x, mamba16H)
        res2 = self.layers[1](res1, mamba16V)
        res3 = self.layers[2](res2, mamba16D)
        w = self.ca(torch.cat([res1, res2, res3], dim=1))
        w = w.view(-1, 3, self.dim)[:, :, :, None, None]
        out = w[:, 0, ::] * res1 + w[:, 1, ::] * res2 + w[:, 2, ::] * res3
        w = self.sa(out)
        out = w * out
        return out + x


class BasicBlock(nn.Module):
    def __init__(self, dim, num_blocks=10):
        super().__init__()
        self.blocks = nn.Sequential(*[TambaBlock(dim) for _ in range(num_blocks)])

    def forward(self, x, mamba16H, mamba16V, mamba16D):
        for block in self.blocks:
            x = block(x, mamba16H, mamba16V, mamba16D)
        return x


class C4RSD(nn.Module):
    def __init__(self, pretrained=True, return_coeffs=False):
        super(C4RSD, self).__init__()
        self.encoder = build_res2net101_encoder(pretrained=pretrained, verbose=False)
        self.decoder = decoder(res_blocks=18)
        self.mambaH = mamba_Branches(scan_mode='H')
        self.mambaV = mamba_Branches(scan_mode='V')
        self.mambaD = mamba_Branches(scan_mode='D')
        self.return_coeffs = return_coeffs

    def check_image_size(self, x):
        # NOTE: for I2I test
        _, _, h, w = x.size()
        base_number = 32
        mod_pad_h = (base_number - h % base_number) % base_number
        mod_pad_w = (base_number - w % base_number) % base_number
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, x):
        _, _, h, w = x.size()
        x = self.check_image_size(x)
        _, cH, cV, cD,_ = HaarWavelet.wavelet_transform_2d(x)

        encoder_out=self.encoder(x)
        pH, outH = self.mambaH(cH)
        pV, outV = self.mambaV(cV)
        pD, outD = self.mambaD(cD)
        x_out = self.decoder(x, encoder_out, pH, pV, pD)
        x_out = x_out[:, :, :h, :w]
        if self.return_coeffs:
            return x_out, (outH, outV, outD)
        return x_out

if __name__ == "__main__":
    from pyzjr.nn import summary_1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = C4RSD()
    model = model.to(device)
    inputs = torch.randn(1, 3, 256, 256).to(device)
    output = model(inputs)
    print("output", output.shape)
    summary_1(model, input_size=(3, 256, 256))