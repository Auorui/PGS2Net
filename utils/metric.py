import torch
import torch.nn.functional as F
from utils.pytorch_ssim import ssim
import lpips
import cv2
import numpy as np

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
        try:
            import pyiqa
        except ImportError:
            raise ImportError(
                "❌ 错误：计算图像质量指标需要安装 `pyiqa` 库。\n"
                "请运行以下命令安装：pip install pyiqa"
            )
        self.psnr_fun = pyiqa.create_metric('psnr', device=device)
        self.ssim_fun = pyiqa.create_metric('ssim', device=device)
        self.lpips_fun = pyiqa.create_metric('lpips', device=device)

    def get_psnr(self):
        return self.psnr_fun(self.output, self.target).item()

    def get_ssim(self):
        return self.ssim_fun(self.output, self.target).item()

    def get_lpips(self):
        return self.lpips_fun(self.output, self.target).item()


class DehazeMetricV3():
    """按论文 / SwinIR 官方口径计算 PSNR / SSIM / LPIPS。

    - PSNR / SSIM：SwinIR 官方实现（含 `test_y_channel` 开关）
    - LPIPS：`lpips` 官方库，`net='alex'` 或 `'vgg'`
    - 输入：output/target 为 [-1, 1] 的 torch.Tensor，形状 (N, 3, H, W)

    Args:
        output (Tensor): 去雾结果，(N, 3, H, W)，值域 [-1, 1]。
        target (Tensor): GT， (N, 3, H, W)，值域 [-1, 1]。
        device (str): 'cuda' 或 'cpu'。
        net (str): LPIPS backbone，'alex' 或 'vgg'。
        test_y_channel (bool): 是否用 Y 通道计算 PSNR/SSIM，论文通常 True。
        align (int): 若 >0，将短边缩放到该尺寸（SwinIR 官方常用 256/512）。
        bgr_input (bool): 输入是否为 BGR 顺序。若你的 tensor 是 RGB，保持 False。
    """

    def __init__(self, output, target, device='cuda', net='alex',
                 test_y_channel=True, align=0, bgr_input=False):
        # ---------- 1. 保证 4 维 (N, C, H, W) ----------
        if output.ndim == 2:      # (H, W)
            output = output.unsqueeze(0).unsqueeze(0)
        elif output.ndim == 3:    # (C, H, W)
            output = output.unsqueeze(0)
        if target.ndim == 2:
            target = target.unsqueeze(0).unsqueeze(0)
        elif target.ndim == 3:
            target = target.unsqueeze(0)

        # ---------- 2. [-1, 1] → [0, 1] ----------
        output = output * 0.5 + 0.5
        target = target * 0.5 + 0.5

        # ---------- 3. 可选对齐 ----------
        if align > 0:
            output = self._align(output, align)
            target = self._align(target, align)

        # ---------- 4. [0, 1] → [0, 255] uint8 numpy ----------
        output_np = (output.clamp(0, 1) * 255.0).round().byte().cpu().numpy()  # (N, C, H, W)
        target_np = (target.clamp(0, 1) * 255.0).round().byte().cpu().numpy()

        # ---------- 5. (N, C, H, W) → (N, H, W, C) ----------
        output_np = output_np.transpose(0, 2, 3, 1)   # (N, H, W, C)
        target_np = target_np.transpose(0, 2, 3, 1)

        # ---------- 6. 单通道 → 复制成 3 通道 ----------
        if output_np.shape[-1] == 1:
            output_np = np.repeat(output_np, 3, axis=-1)
        if target_np.shape[-1] == 1:
            target_np = np.repeat(target_np, 3, axis=-1)

        # ---------- 7. RGB → BGR（SwinIR 的 bgr2ycbcr 假设 BGR）----------
        if not bgr_input:
            output_np = output_np[..., ::-1].copy()   # RGB → BGR
            target_np = target_np[..., ::-1].copy()

        self.output_np = output_np       # (N, H, W, 3) BGR, [0, 255] uint8
        self.target_np = target_np

        # ---------- 8. LPIPS 用的 [0, 1] float tensor ----------
        self.output_lpips = output.clamp(0, 1).to(device)
        self.target_lpips = target.clamp(0, 1).to(device)

        self.device = device
        self.net = net
        self.test_y_channel = test_y_channel

    @staticmethod
    def _align(tensor, size):
        """短边缩放到 size，保持长宽比。"""
        _, _, h, w = tensor.shape
        if min(h, w) == size:
            return tensor
        scale = size / min(h, w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        return F.interpolate(tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)

    # --------------------------------------------------------
    # SwinIR 官方实现（内嵌）
    # --------------------------------------------------------
    @staticmethod
    def _calculate_psnr(img1, img2, test_y_channel=False):
        assert img1.shape == img2.shape
        assert img1.shape[2] == 3
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        if test_y_channel:
            img1 = DehazeMetricV3._to_y_channel(img1)
            img2 = DehazeMetricV3._to_y_channel(img2)
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20. * np.log10(255. / np.sqrt(mse))

    @staticmethod
    def _ssim_single_channel(img1, img2):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())
        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    @staticmethod
    def _calculate_ssim(img1, img2, test_y_channel=False):
        assert img1.shape == img2.shape
        assert img1.shape[2] == 3
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        if test_y_channel:
            img1 = DehazeMetricV3._to_y_channel(img1)
            img2 = DehazeMetricV3._to_y_channel(img2)
        ssims = []
        for i in range(img1.shape[2]):
            ssims.append(DehazeMetricV3._ssim_single_channel(img1[..., i], img2[..., i]))
        return float(np.array(ssims).mean())

    @staticmethod
    def _to_y_channel(img):
        img = img.astype(np.float32) / 255.
        if img.ndim == 3 and img.shape[2] == 3:
            img = DehazeMetricV3._bgr2ycbcr(img, y_only=True)
            img = img[..., None]
        return img * 255.

    @staticmethod
    def _convert_input_type_range(img):
        img_type = img.dtype
        img = img.astype(np.float32)
        if img_type == np.float32:
            pass
        elif img_type == np.uint8:
            img /= 255.
        else:
            raise TypeError(f'img type must be float32 or uint8, got {img_type}')
        return img

    @staticmethod
    def _convert_output_type_range(img, dst_type):
        if dst_type not in (np.uint8, np.float32):
            raise TypeError(f'dst_type must be float32 or uint8, got {dst_type}')
        if dst_type == np.uint8:
            img = img.round()
        else:
            img /= 255.
        return img.astype(dst_type)

    @staticmethod
    def _bgr2ycbcr(img, y_only=False):
        img_type = img.dtype
        img = DehazeMetricV3._convert_input_type_range(img)
        if y_only:
            out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
        else:
            out_img = np.matmul(
                img,
                [[24.966, 112.0, -18.214],
                 [128.553, -74.203, -93.786],
                 [65.481, -37.797, 112.0]]
            ) + [16, 128, 128]
        out_img = DehazeMetricV3._convert_output_type_range(out_img, img_type)
        return out_img

    # --------------------------------------------------------
    # 对外接口
    # --------------------------------------------------------
    def get_psnr(self):
        vals = []
        for i in range(self.output_np.shape[0]):
            vals.append(self._calculate_psnr(
                self.output_np[i], self.target_np[i], test_y_channel=self.test_y_channel))
        return float(np.mean(vals))

    def get_ssim(self):
        vals = []
        for i in range(self.output_np.shape[0]):
            vals.append(self._calculate_ssim(
                self.output_np[i], self.target_np[i], test_y_channel=self.test_y_channel))
        return float(np.mean(vals))

    def get_lpips(self):
        lpips_fn = get_lpips_model(net=self.net, device=self.device)
        with torch.no_grad():
            score = lpips_fn.forward(self.output_lpips, self.target_lpips, normalize=True)
        return score.mean().item()

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
