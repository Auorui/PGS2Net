import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_dct import dct_2d, idct_2d
from models.pgs2net.efficient_dct import DCT2D

# Frequency-Adaptive Spectral Modulation (FASM)
class FASM(nn.Module):
    """Radial Prior Filter (RPF) + Group-wise Frequency Attention (GFA)
    RPF:  M(r) = exp(-alpha * phi(k, r0, r))
          phi 由 act_type 决定:
            'tanh'      : tanh(k(r0-r))      反对称, 低频抑制+高频增强, 有界
            'piecewise' : 单边      仅高频增强, 低频不变 (M=1)
            'sigmoid'   : 2*sigmoid(k(r0-r))-1  反对称, 过渡更软
            'linear'    : k(r0-r)            线性无饱和 (反衬有界性)
          - r0 (低频/高频分界): 跨数据集拟合 ≈ 0.50  (r0 = 0.499 ± 0.017)
          - k  (过渡陡度)     : 跨数据集拟合 ≈ 2.22  (k = 2.06 ± 0.36)
          - alpha (调制强度)   : 可学习, 由训练自适应 (吸收雾浓度差异)
    """
    def __init__(self, in_channels, out_channels, groups=1,
                 use_rpf=False, act_type='tanh',
                 learn_alpha=True, learn_k=True, learn_r0=True,
                 K_INIT=2.22, A_INIT=0.5, R0_INIT=0.5):
        super().__init__()
        self.groups = groups
        self.use_rpf = use_rpf
        self.act_type = act_type
        if use_rpf:
            # 三个参数的可学习性独立控制, 便于参数消融
            self.alpha = nn.Parameter(torch.tensor(float(A_INIT)),  requires_grad=learn_alpha)
            self.k     = nn.Parameter(torch.tensor(float(K_INIT)),  requires_grad=learn_k)
            self.r0    = nn.Parameter(torch.tensor(float(R0_INIT)), requires_grad=learn_r0)

        # 动态频域注意力 (GFA)
        self.weight = nn.Sequential(
            nn.Conv2d(in_channels, groups, 1),
            nn.Softmax(dim=1)
        )
        self.fdc = nn.Conv2d(in_channels, out_channels * groups, 1, groups=groups, bias=True)
        self.fpe = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels)

        self._cached = {}

    def _radial_grid(self, h, w, device, dtype):
        key = (h, w, device, dtype)
        if key not in self._cached:
            yy = torch.arange(h, device=device, dtype=dtype).view(-1, 1)
            xx = torch.arange(w, device=device, dtype=dtype).view(1, -1)
            max_d = ((h - 1) ** 2 + (w - 1) ** 2) ** 0.5
            r = torch.sqrt(xx ** 2 + yy ** 2) / (max_d + 1e-8)  # DCT 低频在 (0,0), 无需 shift
            self._cached[key] = r
        return self._cached[key]

    def _shape_fn(self, r, k, r0):
        """调制形状函数 phi(r); 代回 M = exp(-alpha * phi)"""
        if self.act_type == 'tanh':
            return torch.tanh(k * (r0 - r))                 # 反对称有界
        elif self.act_type == 'sigmoid':
            return 2.0 * torch.sigmoid(k * (r0 - r)) - 1.0  # 反对称, 过渡更软
        elif self.act_type == 'linear':
            return k * (r0 - r)                             # 线性无饱和 (无界)
        elif self.act_type == 'piecewise':
            # 仅高频(r>r0)增强, 低频(r<=r0)保持 M=1
            # phi = -(r-r0)*k for r>r0 -> M = exp(alpha*k*(r-r0)) > 1
            return torch.where(r > r0, -(r - r0) * k, torch.zeros_like(r))
        else:
            raise ValueError(f"unknown act_type: {self.act_type}")

    def radial_modulation(self, x_dct):
        h, w = x_dct.shape[-2], x_dct.shape[-1]
        r = self._radial_grid(h, w, x_dct.device, x_dct.dtype)
        r0 = torch.clamp(self.r0, 0.05, 0.7)  # 约束在合理频带, 防训练漂移
        k = torch.clamp(self.k, 0.1, 50.0)
        M = torch.exp(-self.alpha * self._shape_fn(r, k, r0))
        return M.unsqueeze(0).unsqueeze(0)  # (1,1,H,W) 广播到 (B,C,H,W)

    def forward(self, x):
        B, C, H, W = x.size()
        dct_feat = dct_2d(x, norm='ortho')

        if self.use_rpf:
            dct_feat = dct_feat * self.radial_modulation(dct_feat)

        dct_dyn = self.fpe(dct_feat) + dct_feat                 # 局部频域增强 + 残差
        dy_weight = self.weight(dct_dyn)                        # (B, g, H, W) 频域注意力
        y = self.fdc(dct_dyn).view(B, self.groups, -1, H, W)    # (B, g, Cout/g, H, W)
        y = torch.einsum("bgchw, bghw -> bchw", y, dy_weight)   # group-wise 加权

        fuse = y + dct_feat                                     # 残差
        return idct_2d(fuse, norm='ortho')


# Efficient Frequency-Adaptive Spectral Modulation (EFASM)
# 自己实现的DCT2D可替换torch_dct, 推理速度有所提升
class EFASM(nn.Module):
    """Radial Prior Filter (RPF) + Group-wise Frequency Attention (GFA)
    RPF:  M(r) = exp(-alpha * phi(k, r0, r))
          phi 由 act_type 决定:
            'tanh'      : tanh(k(r0-r))      反对称, 低频抑制+高频增强, 有界
            'piecewise' : 单边      仅高频增强, 低频不变 (M=1)
            'sigmoid'   : 2*sigmoid(k(r0-r))-1  反对称, 过渡更软
            'linear'    : k(r0-r)            线性无饱和 (反衬有界性)
          - r0 (低频/高频分界): 跨数据集拟合 ≈ 0.50  (r0 = 0.499 ± 0.017)
          - k  (过渡陡度)     : 跨数据集拟合 ≈ 2.22  (k = 2.06 ± 0.36)
          - alpha (调制强度)   : 可学习, 由训练自适应 (吸收雾浓度差异)
    """
    def __init__(self, in_channels, out_channels, groups=1,
                 use_rpf=False, act_type='tanh',
                 learn_alpha=True, learn_k=True, learn_r0=True,
                 K_INIT=2.22, A_INIT=0.5, R0_INIT=0.5):
        super().__init__()
        self.groups = groups
        self.use_rpf = use_rpf
        self.act_type = act_type
        if use_rpf:
            # 三个参数的可学习性独立控制, 便于参数消融
            self.alpha = nn.Parameter(torch.tensor(float(A_INIT)),  requires_grad=learn_alpha)
            self.k     = nn.Parameter(torch.tensor(float(K_INIT)),  requires_grad=learn_k)
            self.r0    = nn.Parameter(torch.tensor(float(R0_INIT)), requires_grad=learn_r0)

        # 动态频域注意力 (GFA)
        self.weight = nn.Sequential(
            nn.Conv2d(in_channels, groups, 1),
            nn.Softmax(dim=1)
        )
        self.fdc = nn.Conv2d(in_channels, out_channels * groups, 1, groups=groups, bias=True)
        self.fpe = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels)

        self._cached = {}

    def _radial_grid(self, h, w, device, dtype):
        key = (h, w, device, dtype)
        if key not in self._cached:
            yy = torch.arange(h, device=device, dtype=dtype).view(-1, 1)
            xx = torch.arange(w, device=device, dtype=dtype).view(1, -1)
            max_d = ((h - 1) ** 2 + (w - 1) ** 2) ** 0.5
            r = torch.sqrt(xx ** 2 + yy ** 2) / (max_d + 1e-8)  # DCT 低频在 (0,0), 无需 shift
            self._cached[key] = r
        return self._cached[key]

    def _shape_fn(self, r, k, r0):
        """调制形状函数 phi(r); 代回 M = exp(-alpha * phi)"""
        if self.act_type == 'tanh':
            return torch.tanh(k * (r0 - r))                 # 反对称有界
        elif self.act_type == 'sigmoid':
            return 2.0 * torch.sigmoid(k * (r0 - r)) - 1.0  # 反对称, 过渡更软
        elif self.act_type == 'linear':
            return k * (r0 - r)                             # 线性无饱和 (无界)
        elif self.act_type == 'piecewise':
            # 仅高频(r>r0)增强, 低频(r<=r0)保持 M=1
            # phi = -(r-r0)*k for r>r0 -> M = exp(alpha*k*(r-r0)) > 1
            return torch.where(r > r0, -(r - r0) * k, torch.zeros_like(r))
        else:
            raise ValueError(f"unknown act_type: {self.act_type}")

    def radial_modulation(self, x_dct):
        h, w = x_dct.shape[-2], x_dct.shape[-1]
        r = self._radial_grid(h, w, x_dct.device, x_dct.dtype)
        r0 = torch.clamp(self.r0, 0.05, 0.7)  # 约束在合理频带, 防训练漂移
        k = torch.clamp(self.k, 0.1, 50.0)
        M = torch.exp(-self.alpha * self._shape_fn(r, k, r0))
        return M.unsqueeze(0).unsqueeze(0)  # (1,1,H,W) 广播到 (B,C,H,W)

    def forward(self, x):
        B, C, H, W = x.size()
        dct_feat = DCT2D.dct_2d(x, norm='ortho')

        if self.use_rpf:
            dct_feat = dct_feat * self.radial_modulation(dct_feat)

        dct_dyn = self.fpe(dct_feat) + dct_feat                 # 局部频域增强 + 残差
        dy_weight = self.weight(dct_dyn)                        # (B, g, H, W) 频域注意力
        y = self.fdc(dct_dyn).view(B, self.groups, -1, H, W)    # (B, g, Cout/g, H, W)
        y = torch.einsum("bgchw, bghw -> bchw", y, dy_weight)   # group-wise 加权

        fuse = y + dct_feat                                     # 残差
        return DCT2D.idct_2d(fuse, norm='ortho')


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
    def __init__(self, dim, groups=1, use_rpf=False, act_type='tanh',
                 learn_alpha=True, learn_k=True, learn_r0=True):
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
            self.dim * 2, self.dim * 2, groups=groups, use_rpf=use_rpf,
            act_type=act_type, learn_alpha=learn_alpha,
            learn_k=learn_k, learn_r0=learn_r0
        )
        # self.dct_unit = EFASM(
        #     self.dim * 2, self.dim * 2, groups=groups, use_rpf=use_rpf,
        #     act_type=act_type, learn_alpha=learn_alpha,
        #     learn_k=learn_k, learn_r0=learn_r0
        # )

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
            groups,
            use_rpf,
            act_type='tanh',
            learn_alpha=True, learn_k=True, learn_r0=True
    ):
        super(Mixer, self).__init__()
        self.dim = dim
        self.mixer_local = LocalBranch(dim=self.dim)
        self.mixer_gloal = GlobalBranch(
            dim=self.dim, groups=groups, use_rpf=use_rpf, act_type=act_type,
            learn_alpha=learn_alpha, learn_k=learn_k, learn_r0=learn_r0
        )

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
            groups,
            use_rpf,
            act_type='tanh',
            learn_alpha=True, learn_k=True, learn_r0=True
    ):
        super(SSIB, self).__init__()
        self.dim = dim
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.mixer = Mixer(
            dim=self.dim, groups=groups, use_rpf=use_rpf, act_type=act_type,
            learn_alpha=learn_alpha, learn_k=learn_k, learn_r0=learn_r0
        )
        self.smfn = SMFN(dim=self.dim)

        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        copy = x
        x = self.norm1(x)
        x = self.mixer(x)
        x = x * self.beta + copy

        copy = x
        x = self.norm2(x)
        x = self.smfn(x)
        x = x * self.gamma + copy

        return x


class Stage(nn.Module):
    def __init__(
            self,
            depth,
            dim,
            groups,
            use_rpf,
            act_type='tanh',
            learn_alpha=True, learn_k=True, learn_r0=True
    ) -> None:
        super(Stage, self).__init__()
        # 只有 first_block 启用 RPF; 后续 block 关闭 RPF
        self.first_block = SSIB(
            dim, groups=groups, use_rpf=use_rpf, act_type=act_type,
            learn_alpha=learn_alpha, learn_k=learn_k, learn_r0=learn_r0
        )
        self.blocks = nn.Sequential(*[
                SSIB(
                    dim=dim,
                    use_rpf=False,
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
                 embed_dim=(48, 96, 192, 96, 48), depth=(2, 2, 2, 2, 2),
                 use_rpf=(True, True, True), act_type='tanh',
                 learn_alpha=True, learn_k=True, learn_r0=True):
        super(PGS2Net, self).__init__()
        self.patch_size = patch_size
        # RPF 设计选项, 统一作用于所有启用 RPF 的编码器 stage
        rpf_kw = dict(act_type=act_type, learn_alpha=learn_alpha,
                      learn_k=learn_k, learn_r0=learn_r0)

        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans,
                                      embed_dim=embed_dim[0], kernel_size=3)
        self.layer1 = Stage(depth=depth[0], dim=embed_dim[0], groups=2,
                            use_rpf=use_rpf[0], **rpf_kw)
        self.skip1 = nn.Conv2d(embed_dim[1], embed_dim[0], 1)
        self.downsample1 = DownSample(input_dim=embed_dim[0], output_dim=embed_dim[1],)
        self.layer2 = Stage(depth=depth[1], dim=embed_dim[1], groups=2,
                            use_rpf=use_rpf[1], **rpf_kw)
        self.skip2 = nn.Conv2d(embed_dim[2], embed_dim[1], 1)
        self.downsample2 = DownSample(input_dim=embed_dim[1], output_dim=embed_dim[2],)
        self.layer3 = Stage(depth=depth[2], dim=embed_dim[2], groups=2,
                            use_rpf=use_rpf[2], **rpf_kw)
        self.upsample3 = PatchUnEmbed_for_upsample(patch_size=2, embed_dim=embed_dim[2],
                                                   out_dim=embed_dim[3])
        self.layer8 = Stage(depth=depth[3], dim=embed_dim[3], groups=1, use_rpf=False)
        self.upsample4 = PatchUnEmbed_for_upsample(patch_size=2, embed_dim=embed_dim[3],
                                                   out_dim=embed_dim[4])
        self.layer9 = Stage(depth=depth[4], dim=embed_dim[4], groups=1, use_rpf=False)
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
        depth=[2, 2, 4, 2, 2],
        use_rpf=[True, True, True],
        act_type='tanh'
    )

def PGS2Net_b():
    return PGS2Net(
        embed_dim=[32, 64, 128, 64, 32],
        depth=[4, 4, 8, 4, 4],
        use_rpf=[True, True, True],
        act_type='tanh'
    )