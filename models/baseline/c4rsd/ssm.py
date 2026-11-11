import torch
import math
import torch.nn as nn
import torch.nn.functional as F

def selective_scan(x, delta, A, B, C, D):
    """
    Perform selective scan operation on the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (B, L, ED).
        delta (torch.Tensor): Delta tensor of shape (B, L, ED).
        A (torch.Tensor): A tensor of shape (ED, N).
        B (torch.Tensor): B tensor of shape (B, L, N).
        C (torch.Tensor): C tensor of shape (B, L, N).
        D (torch.Tensor): D tensor of shape (ED).

    Returns:
        torch.Tensor: Output tensor of shape (B, L, ED).
    """

    _, L, _ = x.shape

    deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, ED, N)
    deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, ED, N)

    BX = deltaB * x.unsqueeze(-1)  # (B, L, ED, N)

    hs = pscan(deltaA, BX)

    y = (
        hs @ C.unsqueeze(-1)
    ).squeeze()  # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

    y = y + D * x

    return y


def selective_scan_seq(x, delta, A, B, C, D, dim_inner: int, d_state: int):
    """
    Perform selective scan sequence operation on the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (B, L, ED).
        delta (torch.Tensor): Delta tensor of shape (B, L, ED).
        A (torch.Tensor): A tensor of shape (ED, N).
        B (torch.Tensor): B tensor of shape (B, L, N).
        C (torch.Tensor): C tensor of shape (B, L, N).
        D (torch.Tensor): D tensor of shape (ED).
        dim_inner (int): Inner dimension size.
        d_state (int): State dimension size.

    Returns:
        torch.Tensor: Output tensor of shape (B, L, ED).
    """

    _, L, _ = x.shape

    deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, ED, N)
    deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, ED, N)

    BX = deltaB * x.unsqueeze(-1)  # (B, L, ED, N)

    h = torch.zeros(
        x.size(0),
        dim_inner,
        d_state,
        device=deltaA.device,
    )  # (B, ED, N)
    hs = []

    for t in range(0, L):
        h = deltaA[:, t] * h + BX[:, t]
        hs.append(h)

    hs = torch.stack(hs, dim=1)  # (B, L, ED, N)

    # y = (C.unsqueeze(2) * hs).sum(3)
    y = (
        hs @ C.unsqueeze(-1)
    ).squeeze()  # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

    y = y + D * x

    return y


class SSM(nn.Module):
    def __init__(self, in_features, dt_rank: int, dim_inner: int, d_state: int):
        """
        Initializes the SSM module.

        Args:
            in_features (int): The size of the input features.
            dt_rank (int): The rank of the dt projection.
            dim_inner (int): The inner dimension of the dt projection.
            d_state (int): The dimension of the state.

        """
        super().__init__()
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state

        # Linear layer expecting 'in_features' as the input size
        self.deltaBC_layer = nn.Linear(
            in_features, dt_rank + 2 * d_state, bias=False
        )
        self.dt_proj_layer = nn.Linear(dt_rank, dim_inner, bias=True)

        # Defining A_log and D as parameters
        self.A_log = nn.Parameter(
            torch.log(
                torch.arange(1, d_state + 1, dtype=torch.float32).repeat(
                    dim_inner, 1
                )
            )
        )
        self.D = nn.Parameter(torch.ones(dim_inner))

    def forward(self, x, pscan: bool = True):
        """
        Performs forward pass of the SSM module.

        Args:
            x (torch.Tensor): The input tensor.
            pscan (bool, optional): Whether to use selective_scan or selective_scan_seq. Defaults to True.

        Returns:
            torch.Tensor: The output tensor.

        """
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        deltaBC = self.deltaBC_layer(x)
        delta, B, C = torch.split(
            deltaBC, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj_layer(delta))

        # Assuming selective_scan and selective_scan_seq are defined functions
        if pscan:
            y = selective_scan(x, delta, A, B, C, D)
        else:
            y = selective_scan_seq(x, delta, A, B, C, D)

        return y


class PScan(torch.autograd.Function):
    """

    An implementation of the parallel scan operation in PyTorch (Blelloch version).
    This code is based on Francois Fleuret’s pscan (all credits to him). However, the keys differences are :
    -it has been written in an iterative way (rather than recursive)
    -the backward pass has been rewritten

    Please see docs/pscan.ipynb for a detailed explanation of what happens here.

    Example:
    pscan = PScan.apply

    x = torch.randn(2, 3, 4, 5, requires_grad=True)
    y = torch.randn(2, 3, 4, 5, requires_grad=True)

    model = pscan(x, y)
    model.sum().backward()
    print(x.grad)
    print(y.grad)

    """

    @staticmethod
    def pscan(A, X):
        # A : (B, D, L, N)
        # X : (B, D, L, N)

        # modifies X in place by doing a parallel scan.
        # more formally, X will be populated by these values :
        # H[t] = A[t] * H[t-1] + X[t] with H[0] = 0
        # which are computed in parallel (2*log2(T) sequential steps (ideally), instead of T sequential steps)

        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        # up sweep or reduction step
        Aa = A
        Xa = X
        for k in range(num_steps):
            T = 2 * (Xa.size(2) // 2)

            Aa = Aa[:, :, :T].view(B, D, T // 2, 2, -1)
            Xa = Xa[:, :, :T].view(B, D, T // 2, 2, -1)

            Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 0]))
            Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])

            Aa = Aa[:, :, :, 1]
            Xa = Xa[:, :, :, 1]

        # down sweep
        for k in range(num_steps - 1, -1, -1):
            Aa = A[:, :, 2**k - 1 : L : 2**k]
            Xa = X[:, :, 2**k - 1 : L : 2**k]

            T = 2 * (Xa.size(2) // 2)

            if T < Xa.size(2):
                Xa[:, :, -1].add_(Aa[:, :, -1].mul(Xa[:, :, -2]))
                Aa[:, :, -1].mul_(Aa[:, :, -2])

            Aa = Aa[:, :, :T].view(B, D, T // 2, 2, -1)
            Xa = Xa[:, :, :T].view(B, D, T // 2, 2, -1)

            Xa[:, :, 1:, 0].add_(Aa[:, :, 1:, 0].mul(Xa[:, :, :-1, 1]))
            Aa[:, :, 1:, 0].mul_(Aa[:, :, :-1, 1])

    @staticmethod
    def forward(ctx, A_in, X_in):
        """
        Applies the parallel scan operation, as defined above. Returns a new tensor.

        Args:
            A_in : (B, L, D, N)
            X_in : (B, L, D, N)

        Returns:
            H : (B, L, D, N)
        """

        # clone tensor (in-place ops)
        A = A_in.clone()  # (B, L, D, N)
        X = X_in.clone()  # (B, L, D, N)

        # prepare tensors
        A = A.transpose(2, 1)  # (B, D, L, N)
        X = X.transpose(2, 1)  # (B, D, L, N)

        # parallel scan
        PScan.pscan(A, X)

        ctx.save_for_backward(A_in, X)

        return X.transpose(2, 1)

    @staticmethod
    def backward(ctx, grad_output_in):
        """
        Flows the gradient from the output to the input. Returns two new tensors.

        Args:
            ctx : A_in : (B, L, D, N), X : (B, D, L, N)
            grad_output_in : (B, L, D, N)

        Returns:
            gradA : (B, L, D, N), gradX : (B, L, D, N)
        """

        A_in, X = ctx.saved_tensors

        # clone tensors
        A = A_in.clone()
        # grad_output_in will be cloned with flip()

        # prepare tensors
        A = A.transpose(2, 1)  # (B, D, L, N)
        A = torch.cat((A[:, :, :1], A[:, :, 1:].flip(2)), dim=2)
        grad_output_b = grad_output_in.transpose(2, 1)

        # reverse parallel scan
        grad_output_b = grad_output_b.flip(2)
        PScan.pscan(A, grad_output_b)
        grad_output_b = grad_output_b.flip(2)

        Q = torch.zeros_like(X)
        Q[:, :, 1:].add_(X[:, :, :-1] * grad_output_b[:, :, 1:])

        return Q.transpose(2, 1), grad_output_b.transpose(2, 1)


pscan = PScan.apply