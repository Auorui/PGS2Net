"""VisionMambaBlock module."""

import torch
from einops import rearrange, repeat
from torch import nn, Tensor
from models.baseline.c4rsd.ssm import SSM


# Pair
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class VisionEncoderMambaBlock(nn.Module):
    """
    VisionMambaBlock is a module that implements the Mamba block from the paper
    Vision Mamba: Efficient Visual Representation Learning with Bidirectional
    State Space Model

    Args:
        dim (int): The input dimension of the input tensor.
        dt_rank (int): The rank of the state space model.
        dim_inner (int): The dimension of the inner layer of the
            multi-head attention.
        d_state (int): The dimension of the state space model.


    Example:
    >>> block = VisionMambaBlock(dim=256, heads=8, dt_rank=32,
            dim_inner=512, d_state=256)
    >>> x = torch.randn(1, 32, 256)
    >>> out = block(x)
    >>> out.shape
    torch.Size([1, 32, 256])
    """

    def __init__(
        self,
        dim: int,
        dt_rank: int,
        dim_inner: int,
        d_state: int,
    ):
        super().__init__()
        self.dim = dim
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state

        self.forward_conv1d = nn.Conv1d(
            in_channels=dim, out_channels=dim, kernel_size=1
        )
        self.backward_conv1d = nn.Conv1d(
            in_channels=dim, out_channels=dim, kernel_size=1
        )
        self.norm = nn.LayerNorm(dim)
        self.silu = nn.SiLU()
        self.ssm = SSM(dim, dt_rank, dim_inner, d_state)

        # Linear layer for z and x
        self.proj = nn.Linear(dim, dim)

        # Softplus
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor):
        b, s, d = x.shape

        # Skip connection
        skip = x
        # print(f"x device: {x.device}")
        # print(f"Weight device: {self.norm.weight.device}")  # 如果 self.norm 有权重

        # Normalization
        x = self.norm(x)

        # Split x into x1 and x2 with linears
        z1 = self.proj(x)
        x = self.proj(x)

        # forward con1d
        x1 = self.process_forward(
            x,
            self.forward_conv1d,
            self.ssm,
        )

        # # backward conv1d
        x2 = self.process_backward(
            x,
            self.backward_conv1d,
            self.ssm,
        )

        # Activation
        z = self.silu(z1)

        # Matmul
        x1 *= z
        x2 *= z

        # Residual connection
        return x1 + x2 + skip
        # return x1 + skip

    def process_forward(
        self,
        x: Tensor,
        conv1d: nn.Conv1d,
        ssm: SSM,
    ):
        x = rearrange(x, "b s d -> b d s")
        x = self.softplus(conv1d(x))
        # print(f"Conv1d: {x.shape}")
        x = rearrange(x, "b d s -> b s d")
        x = ssm(x)
        return x
    def process_backward(
        self,
        x: Tensor,
        conv1d: nn.Conv1d,
        ssm: SSM,
    ):
        x = torch.flip(x,dims=[1])
        x = rearrange(x, "b s d -> b d s")
        x = self.softplus(conv1d(x))
        # print(f"Conv1d: {x.shape}")
        x = rearrange(x, "b d s -> b s d")
        x = ssm(x)
        x = torch.flip(x,dims=[1])
        return x

class Vim(nn.Module):
    """
    Vision Mamba (Vim) model implementation.

    Args:
        dim (int): Dimension of the model.
        dt_rank (int, optional): Rank of the dynamic tensor. Defaults to 32.
        dim_inner (int, optional): Inner dimension of the model. Defaults to None.
        d_state (int, optional): State dimension of the model. Defaults to None.
        num_classes (int, optional): Number of output classes. Defaults to None.
        image_size (int, optional): Size of the input image. Defaults to 224.
        patch_size (int, optional): Size of the image patch. Defaults to 16.
        channels (int, optional): Number of image channels. Defaults to 3.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
        depth (int, optional): Number of encoder layers. Defaults to 12.

    Attributes:
        dim (int): Dimension of the model.
        dt_rank (int): Rank of the dynamic tensor.
        dim_inner (int): Inner dimension of the model.
        d_state (int): State dimension of the model.
        num_classes (int): Number of output classes.
        image_size (int): Size of the input image.
        patch_size (int): Size of the image patch.
        channels (int): Number of image channels.
        dropout (float): Dropout rate.
        depth (int): Number of encoder layers.
        to_patch_embedding (nn.Sequential): Sequential module for patch embedding.
        dropout (nn.Dropout): Dropout module.
        cls_token (nn.Parameter): Class token parameter.
        to_latent (nn.Identity): Identity module for latent representation.
        layers (nn.ModuleList): List of encoder layers.
        output_head (output_head): Output head module.

    """

    def __init__(
        self,
        dim: int,
        dt_rank: int = 32,
        d_state: int = None,
        dropout: float = 0.1,
        depth: int = 12,
        scan_mode: str = 'H',
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.dt_rank = dt_rank
        self.d_state = d_state
        self.dropout = dropout
        self.depth = depth

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # encoder layers
        self.layers = nn.ModuleList()

        # Append the encoder layers
        for _ in range(depth):
            self.layers.append(
                VisionEncoderMambaBlock(
                    dim=dim,
                    dt_rank=dt_rank,
                    dim_inner=dim,
                    d_state=d_state,
                    *args,
                    **kwargs,
                )
            )
        # 定义扫描操作映射
        self.scan_mode = scan_mode
        self.scan_ops = {
            'H': self.horizontal_scan,
            'V': self.vertical_scan,
            'D': self.diagonal_scan,  # 假设 diagonal_scan 是全局函数
        }

        # 定义恢复操作映射
        self.recover_ops = {
            'H': self.horizontal_recover,
            'V': self.vertical_recover,
            'D': self.recover_from_diagonal_scan,  # 假设 recover_from_diagonal_scan 是全局函数
        }

    def horizontal_scan(self, x):
        return x.flatten(2, 3).transpose(1, 2)

    def vertical_scan(self, x):
        return x.transpose(2, 3).flatten(2, 3).transpose(1, 2)

    def diagonal_scan(self, x):
        B, C, H, W = x.shape
        
        # 创建所有可能的对角线偏移量
        offsets = torch.arange(-H + 1, W, device=x.device)  # 所有偏移量
        
        diagonals = []
        
        # 提取每个偏移量对应的对角线元素
        for offset in offsets:
            # 使用 diagonal 提取指定 offset 的对角线元素
            diag = x.diagonal(offset=offset, dim1=-2, dim2=-1)  # 在 H 和 W 上操作
            diagonals.append(diag)
        
        result = torch.cat([d for d in diagonals], dim=-1)  # 拼接所有对角线
        
        return result.transpose(1, 2)

    def horizontal_recover(self, x, B, C, H, W):
        return x.transpose(1, 2).reshape(B, C, H, W)

    def vertical_recover(self, x, B, C, H, W):
        return x.transpose(1, 2).reshape(B, C, W, H).transpose(2, 3)
    
    def recover_from_diagonal_scan(self, results, B, C, H, W):
        L = H * W  # 总元素数
        x = torch.zeros(B, C, H, W, dtype=results.dtype, device=results.device)  # 预分配原始形状张量
        
        # 创建所有可能的对角线偏移量
        offsets = torch.arange(-H + 1, W, device=results.device)  # 偏移量从 -H + 1 到 W - 1
        
        # 计算所有偏移量的行列索引
        row_indices = []
        col_indices = []
        
        for offset in offsets:
            if offset < 0:
                row_start = -offset
                row_end = min(H - 1, row_start + W - 1)
            else:
                row_start = 0
                row_end = min(H - 1, W - 1 - offset)
            
            rows = torch.arange(row_start, row_end + 1, device=results.device)
            cols = rows + offset
            row_indices.append(rows)
            col_indices.append(cols)
        
        # 将所有行和列索引拼接成一个张量
        row_indices = torch.cat(row_indices)
        col_indices = torch.cat(col_indices)
        # 直接将结果映射到原始图像
        index = torch.arange(L, device=results.device)  # 直接计算索引
        x[:, :, row_indices, col_indices] = results[:, index, :].transpose(1, 2)
        return x

    def forward(self, x: Tensor):
        # Patch embedding
        B,C,H,W = x.shape
        # print(f"2: {x.shape}")
        x = self.scan_ops[self.scan_mode](x)
        # print(f"3: {x.shape}")
        # Dropout
        x = self.dropout(x)
        
        # Forward pass with the layers
        for layer in self.layers:
            x = layer(x)
        # print(f"4: {x.shape}")
        x = self.recover_ops[self.scan_mode](x, B, self.dim, H, W)
        # print(f"5: {x.shape}")
        return x