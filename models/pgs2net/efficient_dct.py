import torch
import torch.nn as nn

class DCT2D(nn.Module):
    """
    预计算 DCT-II 基矩阵,用 matmul 实现 2D DCT/IDCT。
    对固定尺寸特征图,比 torch_dct 快数倍,结果与 norm='ortho' 一致。
    2D DCT:  X = D_h @ x @ D_w^T
    2D IDCT: x = D_h^T @ X @ D_w
    """
    _cache = {}  # 类变量缓存

    def __init__(self):
        super().__init__()

    @classmethod
    def _dct_matrix(cls, N, device, dtype):
        """类方法，使用类变量缓存"""
        key = (N, device, dtype)
        if key not in cls._cache:
            n = torch.arange(N, device=device, dtype=dtype)
            k = n.view(-1, 1)
            D = torch.cos(torch.pi * (2 * n + 1) * k / (2 * N))
            D[0, :] *= (1.0 / N) ** 0.5
            D[1:, :] *= (2.0 / N) ** 0.5
            cls._cache[key] = D
        return cls._cache[key]

    @staticmethod
    def dct_2d(x, norm=None):
        """2D DCT变换"""
        H, W = x.shape[-2], x.shape[-1]
        Dh = DCT2D._dct_matrix(H, x.device, x.dtype)
        Dw = DCT2D._dct_matrix(W, x.device, x.dtype)
        # X = Dh @ x @ Dw^T
        x = torch.einsum('hi,bcij->bchj', Dh, x)
        x = torch.einsum('bchj,wj->bchw', x, Dw)
        return x

    @staticmethod
    def idct_2d(X, norm=None):
        """2D IDCT变换"""
        H, W = X.shape[-2], X.shape[-1]
        Dh = DCT2D._dct_matrix(H, X.device, X.dtype)
        Dw = DCT2D._dct_matrix(W, X.device, X.dtype)
        # x = Dh^T @ X @ Dw
        X = torch.einsum('ih,bcij->bchj', Dh, X)
        X = torch.einsum('bchj,jw->bchw', X, Dw)
        return X


if __name__ == "__main__":
    import torch_dct

    x = torch.randn(2, 16, 64, 64, device='cuda')
    a = torch_dct.dct_2d(x, norm='ortho')
    b = DCT2D.dct_2d(x)
    print((a - b).abs().max())  # 应该 < 1e-4