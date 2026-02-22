import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.pytorch_ssim import ssim
import lpips
import pyiqa

_lpips_model = None
_lpips_device = None


def get_lpips_model(net='alex', device=None):
    global _lpips_model, _lpips_device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if _lpips_model is None or _lpips_device != device:
        _lpips_model = lpips.LPIPS(net=net, spatial=False, verbose=False)
        _lpips_model = _lpips_model.to(device)
        _lpips_model.eval()
        _lpips_device = device

    return _lpips_model

class DehazeMetricV1():
    def __init__(self, output, target, device='cuda', net='alex'):
        self.output = output * 0.5 + 0.5
        self.target = target * 0.5 + 0.5
        self.net = net
        self.device = device

    def get_psnr(self):
        mse = F.mse_loss(self.output, self.target).to(self.device)
        psnr_val = 10 * torch.log10(1 / mse).item()
        return psnr_val

    def get_ssim(self):
        _, _, H, W = self.output.size()
        down_ratio = max(1, round(min(H, W) / 256))
        ssim_val = ssim(F.adaptive_avg_pool2d(self.output, (int(H / down_ratio), int(W / down_ratio))).to(self.device),
                        F.adaptive_avg_pool2d(self.target, (int(H / down_ratio), int(W / down_ratio))).to(self.device),
                        data_range=1, size_average=False).item()
        return ssim_val

    def get_lpips(self):
        lpips_fn = get_lpips_model(net=self.net, device=self.device)
        # input is [0, 1], normalize = True
        lpips_score = lpips_fn.forward(self.output, self.target, normalize=True)
        return lpips_score.item()

class DehazeMetricV2():
    # https://iqa-pytorch.readthedocs.io/en/latest/ModelCard.html
    def __init__(self, output, target, device='cuda'):
        if output.min() < 0:
            output = (output + 1.0) / 2.0
        if target.min() < 0:
            target = (target + 1.0) / 2.0
        output = output.clamp(0, 1)
        target = target.clamp(0, 1)
        self.output = output
        self.target = target
        self.psnr_fun = pyiqa.create_metric('psnr', device=device)
        self.ssim_fun = pyiqa.create_metric('ssim', device=device)
        self.lpips_fun = pyiqa.create_metric('lpips', device=device)

    def get_psnr(self):
        return self.psnr_fun(self.output, self.target).item()

    def get_ssim(self):
        return self.ssim_fun(self.output, self.target).item()

    def get_lpips(self):
        return self.lpips_fun(self.output, self.target).item()

if __name__ == "__main__":
    output = torch.rand(1, 3, 256, 256).to('cuda') * 2 - 1
    target = torch.rand(1, 3, 256, 256).to('cuda') * 2 - 1
    # print(target.min(), target.max())
    metric_v1 = DehazeMetricV1(output, target)

    psnr_v1 = metric_v1.get_psnr()
    ssim_v1 = metric_v1.get_ssim()
    lpips_v1 = metric_v1.get_lpips()

    print(f"PSNR: {psnr_v1:.4f}")
    print(f"SSIM: {ssim_v1:.4f}")
    print(f"LPIPS: {lpips_v1:.4f}")

    metric_v2 = DehazeMetricV2(output, target)

    psnr_v2 = metric_v2.get_psnr()
    ssim_v2 = metric_v2.get_ssim()
    lpips_v2 = metric_v2.get_lpips()

    print(f"PSNR: {psnr_v1:.4f}")
    print(f"SSIM: {ssim_v1:.4f}")
    print(f"LPIPS: {lpips_v1:.4f}")


