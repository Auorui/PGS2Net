"""Efficient orthonormal 2-D DCT/IDCT implementation for PGS2Net inference.

The transform matrices are generated lazily and cached by spatial size, device,
and data type. This is intended for fixed-resolution deployment, where repeated
inference reuses the same DCT bases and avoids reconstructing them every time.

The implementation follows the orthonormal DCT-II convention used by
``torch_dct.dct_2d(..., norm='ortho')``.

DCT2D is numerically equivalent but not bit-exact to torch_dct, resulting in minimal
differences in inference metrics. For precise reproduction of the paper's numerical results,
the torch_dct path must be enforced. This switch does not involve parameters and the
state_dict is fully compatible.
"""

import torch
import torch.nn as nn

class DCT2D(nn.Module):
    """Matrix-based two-dimensional DCT-II and inverse DCT.

    Forward transform:
        ``X = D_h @ x @ D_w.T``

    Inverse transform:
        ``x = D_h.T @ X @ D_w``

    Notes
    -----
    The cache is shared by all ``DCT2D`` instances. For the fixed 256 x 256 or
    512 x 512 inputs used by PGS2Net, the basis matrices are created during the
    first warm-up pass and reused by all subsequent FASM calls.
    """
    _cache = {}  # Shared cache: (size, device, dtype) -> orthonormal DCT basis.

    def __init__(self):
        super().__init__()

    @classmethod
    def _dct_matrix(cls, N, device, dtype):
        """Return the cached ``N x N`` orthonormal DCT-II basis matrix.

        Parameters
        ----------
        N : int
            Transform length.
        device : torch.device
            Device on which the basis matrix is allocated.
        dtype : torch.dtype
            Data type used by the input feature tensor.
        """
        key = (N, device, dtype)
        if key not in cls._cache:
            n = torch.arange(N, device=device, dtype=dtype)
            k = n.view(-1, 1)
            D = torch.cos(torch.pi * (2 * n + 1) * k / (2 * N))

            # Orthonormal scaling: the DC row uses sqrt(1/N), while all
            # remaining rows use sqrt(2/N).
            D[0, :] *= (1.0 / N) ** 0.5
            D[1:, :] *= (2.0 / N) ** 0.5
            cls._cache[key] = D
        return cls._cache[key]

    @staticmethod
    def dct_2d(x, norm=None):
        """Apply an orthonormal 2-D DCT-II to ``B x C x H x W`` features.

        ``norm`` is accepted for API compatibility with ``torch_dct``. The
        implementation is always orthonormal.
        """
        H, W = x.shape[-2], x.shape[-1]
        Dh = DCT2D._dct_matrix(H, x.device, x.dtype)
        Dw = DCT2D._dct_matrix(W, x.device, x.dtype)

        # X = Dh @ x @ Dw^T
        # Left multiplication transforms rows; right multiplication transforms
        # columns. einsum keeps the batch and channel dimensions independent.
        x = torch.einsum('hi,bcij->bchj', Dh, x)
        x = torch.einsum('bchj,wj->bchw', x, Dw)
        return x

    @staticmethod
    def idct_2d(X, norm=None):
        """Apply the inverse orthonormal 2-D DCT to spectral features."""
        H, W = X.shape[-2], X.shape[-1]
        Dh = DCT2D._dct_matrix(H, X.device, X.dtype)
        Dw = DCT2D._dct_matrix(W, X.device, X.dtype)

        # x = Dh^T @ X @ Dw
        # The inverse of an orthonormal transform is its transpose.
        X = torch.einsum('ih,bcij->bchj', Dh, X)
        X = torch.einsum('bchj,jw->bchw', X, Dw)
        return X


if __name__ == "__main__":
    import torch_dct

    x = torch.randn(2, 16, 64, 64, device='cuda')
    a = torch_dct.dct_2d(x, norm='ortho')
    b = DCT2D.dct_2d(x)
    print((a - b).abs().max())  # Expected maximum error: < 1e-4.
