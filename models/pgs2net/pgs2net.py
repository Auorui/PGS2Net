"""PGS2Net: Physics-Guided Stage-Aware Frequency Spectral Network.

This module implements the restoration network described in the PGS2Net paper.
The architecture combines local spatial encoding with global spectral modeling
for remote-sensing image dehazing.

Main components
---------------
FASM
    Frequency-Adaptive Spectral Modulation, consisting of a physics-guided
    Radial Prior Filter (RPF) and Group-wise Frequency Attention (GFA).
CPA
    Cloud Perception Attention for spatially non-uniform cloud/haze responses.
DPFM
    Dual-Path Feature Mixer that fuses local spatial and global spectral paths.
SMFN
    Serial Multi-Scale Convolutional Feed-forward Network.
SSIB
    Spectral-Spatial Interaction Block used throughout the encoder-decoder.

Notes
-----
The training path uses ``torch_dct``, and the inference path automatically employs the cached ``DCT2D``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_dct import dct_2d, idct_2d
from models.pgs2net.efficient_dct import DCT2D


class FASM(nn.Module):
    """Frequency-Adaptive Spectral Modulation.

    FASM contains two complementary operations:

    1. Radial Prior Filter (RPF): applies a physics-guided radial mask in the
       DCT domain to suppress haze-dominated low-frequency components and
       enhance structure-relevant higher-frequency components.
    2. Group-wise Frequency Attention (GFA): dynamically aggregates grouped
       spectral responses according to the current input feature.

    The radial modulation is defined as

        M(r) = exp(-alpha * phi(k, r0, r)),

    where ``r0`` is the low/high-frequency crossover radius, ``k`` controls
    transition steepness, and ``alpha`` controls modulation strength. The
    default values follow the cross-dataset spectral fitting reported in the
    paper: ``k=2.22`` and ``r0=0.50``.

    Parameters
    ----------
    in_channels : int
        Number of input feature channels.
    out_channels : int
        Number of output feature channels. In PGS2Net this equals
        ``in_channels`` so that the spectral residual can be added directly.
    groups : int, default=1
        Number of frequency-attention groups.
    use_rpf : bool, default=False
        Whether to apply the physics-guided radial prior.
    act_type : {"tanh", "sigmoid", "linear", "piecewise"}
        Shape function used by the radial prior.
    learn_alpha, learn_k, learn_r0 : bool
        Control whether the corresponding RPF parameters are trainable.
    """
    def __init__(self, in_channels, out_channels, groups=1, use_rpf=False, act_type='tanh',
                 learn_alpha=True, learn_k=True, learn_r0=True, K_INIT=2.22, A_INIT=0.5, R0_INIT=0.5):
        super().__init__()
        self.groups = groups
        self.use_rpf = use_rpf
        self.act_type = act_type
        if use_rpf:
            # Independent switches support the parameter-learning ablations.
            self.alpha = nn.Parameter(torch.tensor(float(A_INIT)),  requires_grad=learn_alpha)
            self.k     = nn.Parameter(torch.tensor(float(K_INIT)),  requires_grad=learn_k)
            self.r0    = nn.Parameter(torch.tensor(float(R0_INIT)), requires_grad=learn_r0)

        # GFA predicts a spatially varying mixture weight for each group.
        self.weight = nn.Sequential(
            nn.Conv2d(in_channels, groups, 1),
            nn.Softmax(dim=1)
        )
        self.fdc = nn.Conv2d(in_channels, out_channels * groups, 1, groups=groups, bias=True)
        self.fpe = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels)

        self._cached = {}

    def _radial_grid(self, h, w, device, dtype):
        """Return a cached normalized radial-frequency grid of shape ``H x W``."""
        key = (h, w, device, dtype)
        if key not in self._cached:
            yy = torch.arange(h, device=device, dtype=dtype).view(-1, 1)
            xx = torch.arange(w, device=device, dtype=dtype).view(1, -1)
            max_d = ((h - 1) ** 2 + (w - 1) ** 2) ** 0.5
            # DCT places the DC component at the upper-left corner; no shift is
            # required before computing the distance to (0, 0).
            r = torch.sqrt(xx ** 2 + yy ** 2) / (max_d + 1e-8)
            self._cached[key] = r
        return self._cached[key]

    def _shape_fn(self, r, k, r0):
        """Evaluate the radial shape function ``phi(k, r0, r)``."""
        if self.act_type == 'tanh':
            return torch.tanh(k * (r0 - r))                 # Bounded antisymmetric profile.
        elif self.act_type == 'sigmoid':
            return 2.0 * torch.sigmoid(k * (r0 - r)) - 1.0  # Softer bounded transition.
        elif self.act_type == 'linear':
            return k * (r0 - r)                             # Unbounded linear control.
        elif self.act_type == 'piecewise':
            # Single-sided control: keep low frequencies unchanged and enhance
            # only frequencies above the crossover radius.
            return torch.where(r > r0, -(r - r0) * k, torch.zeros_like(r))
        else:
            raise ValueError(f"unknown act_type: {self.act_type}")

    def radial_modulation(self, x_dct):
        """Construct a positive RPF mask broadcastable to ``B x C x H x W``."""
        h, w = x_dct.shape[-2], x_dct.shape[-1]
        r = self._radial_grid(h, w, x_dct.device, x_dct.dtype)
        # Restrict the fitted parameters to physically meaningful ranges.
        r0 = torch.clamp(self.r0, 0.05, 0.7)
        k = torch.clamp(self.k, 0.1, 50.0)
        M = torch.exp(-self.alpha * self._shape_fn(r, k, r0))
        return M.unsqueeze(0).unsqueeze(0)  # 1 x 1 x H x W

    # -------------------------------------------------------------------------
    # Optional asymmetric-strength RPF (research note; intentionally disabled)
    # -------------------------------------------------------------------------
    # The data simulation and radial spectral statistics reveal a consistent
    # sign pattern: haze amplifies low-frequency energy and attenuates
    # high-frequency energy in the DCT domain. The fitted tanh profile captures
    # the crossover location and the monotonic transition between these two
    # regions. However, the magnitudes of low-frequency amplification and
    # high-frequency attenuation are not necessarily identical.
    #
    # The active formulation above uses one learnable alpha:
    #
    #     M(r) = exp(-alpha * phi(r)).
    #
    # This imposes the same modulation-strength parameter on both sides of r0.
    # A possible extension is to introduce two independently learnable values:
    #
    #     alpha_low  : strength of inverse compensation for r <= r0,
    #                  i.e., suppression of haze-amplified low frequencies;
    #     alpha_high : strength of inverse compensation for r > r0,
    #                  i.e., enhancement of haze-attenuated high frequencies.
    #
    # The corresponding piecewise modulation can be written as
    #
    #     alpha(r) = alpha_low,   r <= r0
    #                alpha_high,  r >  r0
    #     M(r) = exp(-alpha(r) * phi(r)).
    #
    # If alpha_low and alpha_high are initialized to the same value, this
    # extension initially reproduces the current single-alpha formulation and
    # then allows the two compensation strengths to adapt independently.
    # 该版本用于表达低频与高频退化幅度可能不对称，目前未启用。
    #
    # Example implementation (requires retraining and a new ablation study):
    #
    # In __init__:
    #     self.alpha_low = nn.Parameter(
    #         torch.tensor(float(A_INIT)), requires_grad=learn_alpha
    #     )
    #     self.alpha_high = nn.Parameter(
    #         torch.tensor(float(A_INIT)), requires_grad=learn_alpha
    #     )
    #
    # Alternative radial modulation:
    #     def radial_modulation_asymmetric(self, x_dct):
    #         h, w = x_dct.shape[-2], x_dct.shape[-1]
    #         r = self._radial_grid(h, w, x_dct.device, x_dct.dtype)
    #         r0 = torch.clamp(self.r0, 0.05, 0.7)
    #         k = torch.clamp(self.k, 0.1, 50.0)
    #         phi = self._shape_fn(r, k, r0)
    #         alpha_r = torch.where(r <= r0, self.alpha_low, self.alpha_high)
    #         M = torch.exp(-alpha_r * phi)
    #         return M.unsqueeze(0).unsqueeze(0)
    #
    # This block remains commented out because enabling it changes the model
    # parameters and checkpoint structure. The submitted paper and released
    # results continue to use the single-alpha RPF defined above.

    def forward(self, x):
        """Apply DCT-domain prior modulation and adaptive group aggregation."""
        B, C, H, W = x.size()
        # 训练阶段保持原始 torch_dct 路径；eval/inference 阶段自动使用预计算基矩阵的 DCT2D。
        # 该切换不引入任何参数，因此与已有 checkpoint 的 state_dict 完全兼容。
        use_efficient_dct = not self.training
        if use_efficient_dct:
            dct_feat = DCT2D.dct_2d(x, norm='ortho')
        else:
            dct_feat = dct_2d(x, norm='ortho')

        if self.use_rpf:
            dct_feat = dct_feat * self.radial_modulation(dct_feat)

        # Enhance correlations between neighboring DCT coefficients.
        dct_dyn = self.fpe(dct_feat) + dct_feat
        dy_weight = self.weight(dct_dyn)                        # (B, g, H, W)
        y = self.fdc(dct_dyn).view(B, self.groups, -1, H, W)    # (B, g, Cout/g, H, W)
        y = torch.einsum("bgchw, bghw -> bchw", y, dy_weight)   # group-wise 加权

        fuse = y + dct_feat                                     # 残差
        if use_efficient_dct:
            return DCT2D.idct_2d(fuse, norm='ortho')
        return idct_2d(fuse, norm='ortho')



class CPAtten(nn.Module):
    """Cloud Perception Attention (CPA).

    CPA complements spectral modulation with spatially adaptive cloud/haze
    perception. It combines three global channel descriptors - average, maximum,
    and standard deviation - with a learned spatial key/value representation.

    Input and output tensors both have shape ``B x C x H x W``.
    """
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
        # Fuse spatial maps derived from average, maximum, and deviation cues.
        self.m3 = nn.Sequential(
            nn.Conv2d(3, 1, 1, 1),
            nn.BatchNorm2d(1),
            nn.SiLU()
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # GAP
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # GMP
        self.gamma_std = nn.Parameter(torch.ones(1, dim, 1, 1))

    def std_pool(self, x):
        """Return a learnably scaled channel-wise global standard deviation."""
        std_global = torch.std(x, dim=[2, 3], keepdim=True)  # [N, C, 1, 1]
        return std_global * self.gamma_std

    def forward(self, x):
        """Recalibrate features using global cloud-aware statistics."""
        n, c, h, w = x.shape
        # Spatial key distribution and channel-preserving value representation.
        k = self.k(x).view(n, 1, -1, 1).softmax(2)  # N x 1 x HW x 1
        v = self.v(x).view(n, 1, c, -1)             # N x 1 x C x HW

        # Global key-value context: N x C x 1 x 1.
        kv = torch.matmul(v, k).view(n, c, 1, 1)

        # Channel descriptors act as complementary query statistics.
        avg = self.avg_pool(x).softmax(1).view(n, 1, 1, c)
        max = self.max_pool(x).softmax(1).view(n, 1, 1, c)
        std = self.std_pool(x).view(n, 1, 1, c)

        # Project each descriptor back to a spatial response map.
        q_avg = torch.matmul(avg, v).view(n, 1, h, w)
        q_max = torch.matmul(max, v).view(n, 1, h, w)
        q_std = torch.matmul(std, v).view(n, 1, h, w)
        # y_cat:[N, 3, H, W]
        q = torch.cat((q_avg, q_max, q_std), 1)  # N x 3 x H x W
        # Residual cloud-aware recalibration, corresponding to F_out=F+Psi(F).
        y = self.m(kv) * self.m3(q).sigmoid()
        return x + y


class GlobalBranch(nn.Module):
    """Global spectral branch of the Dual-Path Feature Mixer.

    The branch expands the channel dimension, performs FASM in the DCT domain,
    projects the result back to the original width, and finally applies CPA to
    account for spatially non-uniform cloud distributions.
    """
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

    def forward(self, x):
        """Encode global spectral degradation and cloud-aware context."""
        x = self.conv_init(x)
        x0 = x
        x = self.dct_unit(x)
        x = self.conv_fina(x + x0)
        x = self.cpa(x)
        return x


class LocalBranch(nn.Module):
    """Local spatial branch using complementary dilation rates.

    The input channels are split evenly. One half is processed by a standard
    depthwise 3 x 3 convolution and the other by a dilated depthwise 3 x 3
    convolution, preserving local texture and moderately enlarged context.
    """
    def __init__(self, dim):
        super(LocalBranch, self).__init__()
        self.dim = dim
        self.dim_sp = dim//2
        self.conv_d1 = nn.Conv2d(self.dim_sp, self.dim_sp, 3, stride=1, padding=1, dilation=1, groups=self.dim_sp)
        self.conv_d2 = nn.Conv2d(self.dim_sp, self.dim_sp, 3, stride=1, padding=2, dilation=2, groups=self.dim_sp)

    def forward(self, x):
        """Return concatenated local features with unchanged tensor shape."""
        x1, x2 = x.chunk(2, dim=1)
        cd1 = self.conv_d1(x1)
        cd2 = self.conv_d2(x2)
        x = torch.cat([cd1, cd2], dim=1)
        return x


class Mixer(nn.Module):
    """Dual-Path Feature Mixer (DPFM).

    DPFM expands the input and splits it into two equal paths:

    - a local spatial path for edges, textures, and geometric continuity;
    - a global spectral path for DCT-domain degradation modeling.

    Channel attention then recalibrates the concatenated representation before
    a point-wise projection restores the original channel dimension.
    """
    def __init__(self, dim, groups, use_rpf, act_type='tanh',
                 learn_alpha=True, learn_k=True, learn_r0=True):
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
        """Fuse local spatial and global spectral representations."""
        x = self.conv_init(x)
        x = list(torch.split(x, self.dim, dim=1))
        x_local = self.mixer_local(x[0])
        x_gloal = self.mixer_gloal(x[1])
        x = torch.cat([x_local, x_gloal], dim=1)
        x = self.gelu(x)
        x = self.ca(x) * x
        x = self.ca_conv(x)
        return x


class SMFN(nn.Module):
    """Serial Multi-Scale Convolutional Feed-forward Network.

     The receptive field is progressively enlarged using serial depthwise
     convolutions with kernel sizes 3 x 3, 5 x 5, and 7 x 7. Point-wise
     projections expand and restore the channel dimension, while BatchNorm and
     GELU provide normalization and nonlinear transformation.

     This serial arrangement corresponds to Eq. (12) in the paper and differs
     from conventional parallel multi-scale fusion.
     """
    def __init__(self, dim, expansion_ratio=4):
        super(SMFN, self).__init__()
        self.dim = dim
        self.hidden_dim = dim * expansion_ratio
        # Progressive receptive-field enlargement: 3 x 3 -> 5 x 5 -> 7 x 7.
        self.multiscale_conv = nn.Sequential(
            # Local-scale depthwise convolution followed by channel expansion.
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, self.hidden_dim, 1, bias=False),  # 通道扩展
            nn.BatchNorm2d(self.hidden_dim),
            nn.GELU(),
            # Mid-scale transformation.
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 5, padding=2,
                      groups=self.hidden_dim, bias=False),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 1, bias=False),
            nn.BatchNorm2d(self.hidden_dim),
            nn.GELU(),
            # Long-range transformation followed by channel restoration.
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 7, padding=3,
                      groups=self.hidden_dim, bias=False),
            nn.Conv2d(self.hidden_dim, dim, 1, bias=False),  # channel restoration
            nn.BatchNorm2d(dim)
        )
        self.residual_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        """Apply serial multi-scale refinement with a learnable residual scale."""
        identity = x
        x_processed = self.multiscale_conv(x)
        output = self.residual_scale * identity + x_processed
        return output


class SSIB(nn.Module):
    """Spectral-Spatial Interaction Block.

    Each SSIB contains a DPFM followed by an SMFN. Batch normalization is
    applied before each submodule, and learnable zero-initialized residual
    scales (``beta`` and ``gamma``) stabilize optimization.

    Parameters
    ----------
    dim : int
        Feature width of the current stage.
    groups : int
        Number of groups used by GFA in the spectral branch.
    use_rpf : bool
        Whether this block activates the physics-guided radial prior.
    """
    def __init__(self, dim, groups, use_rpf, act_type='tanh',
                 learn_alpha=True, learn_k=True, learn_r0=True):
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
        """Apply spectral-spatial mixing and multi-scale feed-forward refinement."""
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
    """A sequence of SSIBs operating at one spatial resolution.

    Following the stage-aware strategy in the paper, RPF is enabled only in
    the first block of an encoder stage. Remaining blocks retain the learned
    spectral-spatial processing but do not repeatedly impose the radial prior.
    """
    def __init__(self, depth, dim, groups, use_rpf, act_type='tanh',
                 learn_alpha=True, learn_k=True, learn_r0=True) -> None:
        super(Stage, self).__init__()
        # Stage-aware prior injection: only the first block may activate RPF.
        # 避免在同一 stage 内重复施加物理先验, 这样会出现振铃效应
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
        """Process one encoder or decoder stage."""
        input = self.first_block(x)
        output = self.blocks(input)
        return output


class PatchEmbed(nn.Module):
    """Convolutional image-to-feature embedding layer."""
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, bias=True)

    def forward(self, x):
        """Project an image or feature map to the stage embedding space."""
        x = self.proj(x)
        return x


class PatchUnEmbed(nn.Module):
    """Project decoder features to the requested output channels."""
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
        """Convert the final feature map to restoration parameters."""
        x = self.proj(x)
        return x


class PatchUnEmbed_for_upsample(nn.Module):
    """Sub-pixel upsampling layer based on point-wise projection and PixelShuffle."""
    def __init__(self, patch_size=4, embed_dim=96, out_dim=64):
        super().__init__()
        self.embed_dim = embed_dim

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_dim * patch_size ** 2, kernel_size=1, bias=False),
            nn.PixelShuffle(patch_size),
        )

    def forward(self, x):
        """Upsample a feature map by ``patch_size``."""
        x = self.proj(x)
        return x


class DownSample(nn.Module):
    """Reduce spatial resolution by two and double the channel dimension.

    Shape transformation: ``B x C x H x W`` ->
    ``B x 2C x H/2 x W/2``.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = output_dim

        self.proj = nn.Sequential(nn.Conv2d(input_dim, input_dim * 2, kernel_size=2, stride=2))

    def forward(self, x):
        """Apply stride-2 convolutional downsampling."""
        x = self.proj(x)
        return x


class PGS2Net(nn.Module):
    """Physics-Guided Stage-Aware Frequency Spectral Network. (PGS²Net)

    PGS2Net is a symmetric encoder-decoder restoration network constructed from
    SSIBs. The encoder progressively injects the radial spectral prior, while
    the decoder reconstructs spatial details without RPF constraints.

    The output head predicts one multiplicative map ``K`` and a three-channel
    additive map ``B``. The restored image is computed as

        I_out = K * I_in - B + I_in.

    Parameters
    ----------
    in_chans : int, default=3
        Number of input image channels.
    out_chans : int, default=4
        Output channels of the restoration head: one for ``K`` and three for
        ``B``.
    patch_size : int, default=1
        Initial embedding stride.
    embed_dim : sequence of int
        Channel widths for the three encoder and two decoder stages.
    depth : sequence of int
        Number of SSIBs in each stage.
    use_rpf : sequence of bool
        RPF placement switches for the three encoder stages.
    """
    def __init__(self, in_chans=3, out_chans=4, patch_size=1,
                 embed_dim=(48, 96, 192, 96, 48), depth=(2, 2, 2, 2, 2),
                 use_rpf=(True, True, True), act_type='tanh',
                 learn_alpha=True, learn_k=True, learn_r0=True):
        super(PGS2Net, self).__init__()
        self.patch_size = patch_size
        # Shared RPF configuration for all enabled encoder stages.
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
        """Pad the input according to the configured patch embedding stride."""
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        """Run the symmetric encoder-decoder and predict ``K``/``B`` features."""
        x = self.patch_embed(x)

        # Encoder stage 1: full-resolution representation.
        x = self.layer1(x)
        copy1 = x

        # Encoder stage 2: half-resolution representation.
        x = self.downsample1(x)
        x = self.layer2(x)
        copy2 = x

        # Bottleneck stage: quarter-resolution global representation.
        x = self.downsample2(x)
        x = self.layer3(x)
        x = self.upsample3(x)

        # Decoder stage 1 with the corresponding encoder skip feature.
        x = self.skip2(torch.cat([x, copy2], dim=1))
        x = self.layer8(x)
        x = self.upsample4(x)

        # Decoder stage 2 and restoration-parameter prediction.
        x = self.skip1(torch.cat([x, copy1], dim=1))
        x = self.layer9(x)
        x = self.patch_unembed(x)

        return x

    def forward(self, x):
        """Restore a haze-free image while preserving the original image size."""
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        feat = self.forward_features(x)

        # One-channel multiplicative map K and three-channel additive map B.
        K, B = torch.split(feat, (1, 3), dim=1)
        x = K * x - B + x
        x = x[:, :, :H, :W]
        return x

def PGS2Net_s():
    """Build the small PGS2Net configuration reported in the paper."""
    return PGS2Net(
        embed_dim=[24, 48, 96, 48, 24],
        depth=[2, 2, 4, 2, 2],
        use_rpf=[True, True, True],
        act_type='tanh'
    )

def PGS2Net_b():
    """Build the base PGS2Net configuration reported in the paper."""
    return PGS2Net(
        embed_dim=[32, 64, 128, 64, 32],
        depth=[4, 4, 8, 4, 4],
        use_rpf=[True, True, True],
        act_type='tanh'
    )

if __name__ == "__main__":
    from pyzjr.nn import summary_1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PGS2Net_s()
    model = model.to(device)
    inputs = torch.randn(1, 3, 256, 256).to(device)
    output = model(inputs)
    print("output", output.shape)
    summary_1(model, input_size=(3, 256, 256))
