import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_dct import dct_2d, idct_2d

# Frequency-Adaptive Spectral Modulation (FASM)
class FASM(nn.Module):
    """Radial Prior Filter (RPF) + Group-wise Frequency Attention (GFA)"""
    def __init__(self, in_channels, out_channels, groups=1, beta=1.0, filter_type='swish'):
        super().__init__()
        self.groups = groups
        self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float32))
        self.filter_type = filter_type
        # 动态权重预测 -> 注意力
        self.weight = nn.Sequential(
            nn.Conv2d(in_channels, groups, 1),
            nn.Softmax(dim=1)
        )
        # group 两阶段频域卷积
        self.fdc = nn.Conv2d(
            in_channels, out_channels * groups,
            kernel_size=1, groups=groups, bias=True
        )
        # 频域增强残差
        self.fpe = nn.Conv2d(
            in_channels, in_channels, kernel_size=3,
            stride=1, padding=1, groups=in_channels
        )

    def spatial_filter(self, x_dct):
        # 手动空间滤波器 -> 物理先验
        h, w = x_dct.shape[2], x_dct.shape[3]
        y = torch.arange(h, device=x_dct.device).float().view(-1, 1)
        x = torch.arange(w, device=x_dct.device).float().view(1, -1)

        distance = torch.sqrt(x ** 2 + y ** 2)
        max_dist = torch.sqrt(torch.tensor(float((h - 1) ** 2 + (w - 1) ** 2),
                                           device=x_dct.device))
        normalized_dist = distance / max_dist

        if self.filter_type == 'softplus':
            x_val = 4.0 * (normalized_dist - 0.1)
            softplus_enhance = torch.log(1 + torch.exp(self.beta * x_val)) / self.beta
            filter_output = 0.8 + 0.4 * softplus_enhance
            filter_output = torch.clamp(filter_output, 0.8, 2.0)
        else:  # swish
            x_val = 3.0 * (normalized_dist - 0.2)
            swish_enhance = x_val * torch.sigmoid(self.beta * x_val)
            max_enhance = torch.max(torch.abs(swish_enhance)) + 1e-8
            filter_output = 1.0 + 0.8 * swish_enhance / max_enhance
            filter_output = torch.clamp(filter_output, 0.7, 2.2)

        return filter_output.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        B, C, H, W = x.size()
        dct_feat = dct_2d(x, norm='ortho')
        if self.filter_type is not None:
            dct_feat = dct_feat * self.spatial_filter(dct_feat)
        # 动态加权分支
        # 频域残差增强
        dct_dyn = self.fpe(dct_feat) + dct_feat
        # 生成分组权重
        dy_weight = self.weight(dct_dyn)  # (B, g, H, W)
        # 分组频域卷积
        y = self.fdc(dct_dyn).view(B, self.groups, -1, H, W)  # (B, g, Cout/g, H, W)
        y = torch.einsum("bgchw, bghw -> bchw", y, dy_weight)  # group attention
        fuse = y + dct_feat
        out = idct_2d(fuse, norm='ortho')
        return out


# Cloud Perception Attention (CPA)
class CPAtten(nn.Module):
    def __init__(self, dim):
        super(CPAtten, self).__init__()
        self.dim = dim
        self.k = nn.Sequential(
            nn.Conv2d(dim, 1, 1, 1),
            nn.BatchNorm2d(1),
            nn.SiLU()
        )
        self.v = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1),
            nn.BatchNorm2d(dim),
            nn.SiLU()
        )
        self.m = nn.Conv2d(dim, dim, 1, 1)
        # 用于 avg/max/std 的空间调制
        self.m3 = nn.Sequential(
            nn.Conv2d(3, 1, 1, 1),
            nn.BatchNorm2d(1),
            nn.SiLU()
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # GAP
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # GMP
        self.gamma_std = nn.Parameter(torch.ones(1, dim, 1, 1))

    def std_pool(self, x):
        # 计算通道标准差作为云雾密度指标
        std_global = torch.std(x, dim=[2, 3], keepdim=True)  # [N, C, 1, 1]
        return std_global * self.gamma_std

    def forward(self, x):
        n, c, h, w = x.shape
        # 计算Key和Value
        k = self.k(x).view(n, 1, -1, 1).softmax(2) # [N, 1, HW, 1]
        v = self.v(x).view(n, 1, c, -1)  # [N, 1, C, HW]
        # 计算KV: [N, C, 1, 1]
        kv = torch.matmul(v, k).view(n, c, 1, 1)
        # avg max std: [N, C, 1, 1] 全局统计特征 (替代Q)
        avg = self.avg_pool(x).softmax(1).view(n, 1, 1, c)
        max = self.max_pool(x).softmax(1).view(n, 1, 1, c)
        std = self.std_pool(x).view(n, 1, 1, c)
        # q: [N, 1, H, W]
        q_avg = torch.matmul(avg, v).view(n, 1, h, w)
        q_max = torch.matmul(max, v).view(n, 1, h, w)
        q_std = torch.matmul(std, v).view(n, 1, h, w)
        # y_cat:[N, 3, H, W]
        q = torch.cat((q_avg, q_max, q_std), 1)
        # 计算注意力权重 (Q * KV)，并进行标准化
        y = self.m(kv) * self.m3(q).sigmoid()
        return x + y

# GSPM (Global Spectral Perception Modulation)
class GlobalBranch(nn.Module):
    def __init__(self, dim, filter_type, groups=1):
        super(GlobalBranch, self).__init__()
        self.dim = dim
        self.cpa = CPAtten(dim)
        self.conv_init = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1),
            nn.GELU()
        )
        self.conv_fina = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.GELU()
        )
        self.dct_unit = FASM(
            self.dim * 2, self.dim * 2, groups=groups, filter_type=filter_type, beta=1
        )

    def forward(self, x):
        x = self.conv_init(x)
        x0 = x
        x = self.dct_unit(x)
        x = self.conv_fina(x + x0)
        x = self.cpa(x)
        return x

# LSEM (Local Spatial Encoding Module)
class LocalBranch(nn.Module):
    def __init__(self, dim):
        super(LocalBranch, self).__init__()
        self.dim = dim
        self.dim_sp = dim//2
        self.conv_d1 = nn.Conv2d(self.dim_sp, self.dim_sp, 3, stride=1, padding=1, dilation=1, groups=self.dim_sp)
        self.conv_d2 = nn.Conv2d(self.dim_sp, self.dim_sp, 3, stride=1, padding=2, dilation=2, groups=self.dim_sp)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        cd1 = self.conv_d1(x1)
        cd2 = self.conv_d2(x2)
        x = torch.cat([cd1, cd2], dim=1)
        return x

# Dual-Path Feature Mixer (DPFM)
class Mixer(nn.Module):
    def __init__(
            self,
            dim,
            filter_type,
            groups
    ):
        super(Mixer, self).__init__()
        self.dim = dim
        self.mixer_local = LocalBranch(dim=self.dim)
        self.mixer_gloal = GlobalBranch(dim=self.dim, filter_type=filter_type, groups=groups)

        self.ca_conv = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1),
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2 * dim, 2 * dim//2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * dim//2, 2 * dim, kernel_size=1),
            nn.Sigmoid()
        )

        self.gelu = nn.GELU()
        self.conv_init = nn.Sequential(
            nn.Conv2d(dim, 2 * dim, 1),
        )

    def forward(self, x):
        x = self.conv_init(x)
        x = list(torch.split(x, self.dim, dim=1))
        x_local = self.mixer_local(x[0])
        x_gloal = self.mixer_gloal(x[1])
        x = torch.cat([x_local, x_gloal], dim=1)
        x = self.gelu(x)
        x = self.ca(x) * x
        x = self.ca_conv(x)
        return x

# Serial Multi-Scale Conv Feed-forward Network (SMFN)
class SMFN(nn.Module):
    def __init__(self, dim, expansion_ratio=4):
        super(SMFN, self).__init__()
        self.dim = dim
        self.hidden_dim = dim * expansion_ratio
        # 主分支：3x3 -> 5x5 -> 7x7 深度可分离卷积
        self.multiscale_conv = nn.Sequential(
            # 3x3 深度可分离卷积 + 通道扩展
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, self.hidden_dim, 1, bias=False),  # 通道扩展
            nn.BatchNorm2d(self.hidden_dim),
            nn.GELU(),
            # 5x5 深度可分离卷积
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 5, padding=2,
                      groups=self.hidden_dim, bias=False),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 1, bias=False),
            nn.BatchNorm2d(self.hidden_dim),
            nn.GELU(),
            # 7x7 深度可分离卷积 + 通道恢复
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 7, padding=3,
                      groups=self.hidden_dim, bias=False),
            nn.Conv2d(self.hidden_dim, dim, 1, bias=False),  # 恢复原始通道
            nn.BatchNorm2d(dim)
        )
        self.residual_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        identity = x
        # 多核卷积处理增强后的特征
        x_processed = self.multiscale_conv(x)
        output = self.residual_scale * identity + x_processed
        return output


# Spectral–Spatial Interaction Block (SSIB)
class SSIB(nn.Module):
    def __init__(
            self,
            dim,
            filter_type,
            groups
    ):
        super(SSIB, self).__init__()
        self.dim = dim
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.mixer = Mixer(dim=self.dim, filter_type=filter_type, groups=groups)
        self.smkb = SMFN(dim=self.dim)

        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        copy = x
        x = self.norm1(x)
        x = self.mixer(x)
        x = x * self.beta + copy

        copy = x
        x = self.norm2(x)
        x = self.smkb(x)
        x = x * self.gamma + copy

        return x


class Stage(nn.Module):
    def __init__(
            self,
            depth,
            dim,
            filter_type,
            groups=1
    ) -> None:
        super(Stage, self).__init__()
        # Init blocks
        self.first_block = SSIB(dim, filter_type=filter_type, groups=groups)
        self.blocks = nn.Sequential(*[
                SSIB(
                    dim=dim,
                    filter_type=None,
                    groups=1
                )
            for index in range(depth - 1)
        ])

    def forward(self, x):
        input = self.first_block(x)
        output = self.blocks(input)
        return output


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, bias=True)

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=kernel_size,
                      padding=kernel_size // 2, bias=True),
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchUnEmbed_for_upsample(nn.Module):
    def __init__(self, patch_size=4, embed_dim=96, out_dim=64):
        super().__init__()
        self.embed_dim = embed_dim

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_dim * patch_size ** 2, kernel_size=1, bias=False),
            nn.PixelShuffle(patch_size),
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class DownSample(nn.Module):
    """
    DownSample: Conv
    B*H*W*C -> B*(H/2)*(W/2)*(2*C)
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = output_dim

        self.proj = nn.Sequential(nn.Conv2d(input_dim, input_dim * 2, kernel_size=2, stride=2))

    def forward(self, x):
        x = self.proj(x)
        return x

# A Physics-Guided Stage-Aware Frequency Spectral Network for Remote Sensing Image Dehazing（PGS²-Net）
class PGS2Net(nn.Module):
    def __init__(self, in_chans=3, out_chans=4, patch_size=1,
                 embed_dim=(48, 96, 192, 96, 48), depth=(2, 2, 2, 2, 2)):
        super(PGS2Net, self).__init__()
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans,
                                      embed_dim=embed_dim[0], kernel_size=3)
        self.layer1 = Stage(depth=depth[0], dim=embed_dim[0], filter_type='softplus', groups=2)
        self.skip1 = nn.Conv2d(embed_dim[1], embed_dim[0], 1)
        self.downsample1 = DownSample(input_dim=embed_dim[0], output_dim=embed_dim[1],)
        self.layer2 = Stage(depth=depth[1], dim=embed_dim[1], filter_type='swish', groups=2)
        self.skip2 = nn.Conv2d(embed_dim[2], embed_dim[1], 1)
        self.downsample2 = DownSample(input_dim=embed_dim[1], output_dim=embed_dim[2],)
        self.layer3 = Stage(depth=depth[2], dim=embed_dim[2], filter_type=None)
        self.upsample3 = PatchUnEmbed_for_upsample(patch_size=2, embed_dim=embed_dim[2],
                                                   out_dim=embed_dim[3])
        self.layer8 = Stage(depth=depth[3], dim=embed_dim[3], filter_type=None)
        self.upsample4 = PatchUnEmbed_for_upsample(patch_size=2, embed_dim=embed_dim[3],
                                                   out_dim=embed_dim[4])
        self.layer9 = Stage(depth=depth[4], dim=embed_dim[4], filter_type=None)
        self.patch_unembed = PatchUnEmbed(patch_size=patch_size, out_chans=out_chans,
                                          embed_dim=embed_dim[4], kernel_size=3)

    def check_image_size(self, x):
        # NOTE: for I2I test
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.layer1(x)
        copy1 = x

        x = self.downsample1(x)
        x = self.layer2(x)
        copy2 = x

        x = self.downsample2(x)
        x = self.layer3(x)
        x = self.upsample3(x)

        x = self.skip2(torch.cat([x, copy2], dim=1))
        x = self.layer8(x)
        x = self.upsample4(x)

        x = self.skip1(torch.cat([x, copy1], dim=1))
        x = self.layer9(x)
        x = self.patch_unembed(x)

        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        feat = self.forward_features(x)
        K, B = torch.split(feat, (1, 3), dim=1)
        x = K * x - B + x
        x = x[:, :, :H, :W]
        return x


def PGS2Net_s():
    return PGS2Net(
        embed_dim=[24, 48, 96, 48, 24],
        depth=[2, 2, 4, 2, 2])

def PGS2Net_b():
    return PGS2Net(
        embed_dim=[32, 64, 128, 64, 32],
        depth=[4, 4, 8, 4, 4]
    )

if __name__ == "__main__":
    from ptflops import get_model_complexity_info, flops_counter
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    from calflops import calculate_flops
    from pyzjr import summary_2, model_complexity_info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PGS2Net_s()
    model = model.to(device)
    inputs = torch.randn(1, 3, 256, 256).to(device)
    output = model(inputs)
    print("output", output.shape)
    # summary_2(model, (3, 256, 256))
    model_complexity_info(model, (3, 256, 256))
    # flops, macs, param = calculate_flops(
    #     model=model,
    #     input_shape=(1, 3, 256, 256)
    # )
    # print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, param))
    # flops = FlopCountAnalysis(model, inputs)
    # print(f"Total FLOPs: {flops.total() / 1e9:.3f} G")

    # macs, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True,
    #                                          print_per_layer_stat=False, verbose=False)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # # MACs: 21.572 GMac
    # # FLOPs: 43.144 GFLOPs
    # # Params: 2.338 M


    # FLOPs:42.76 GFLOPS   MACs:21.05 GMACs   Params:2.34 M
