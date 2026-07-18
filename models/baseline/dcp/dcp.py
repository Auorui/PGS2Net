# https://blog.csdn.net/m0_62919535/article/details/148291799
import torch
import torch.nn as nn

class DCP(nn.Module):
    def __init__(self, omega=0.95, t0=0.1, top_percent=0.001, radius=40):
        super(DCP, self).__init__()
        self.omega = omega
        self.t0 = t0
        self.top_percent = top_percent  # 用于估计大气光的像素百分比
        self.radius = radius

    def dark_channel(self, img):
        """计算暗通道 (B, C, H, W) -> (B, H, W)"""
        return torch.min(img, dim=1)[0]  # 取RGB通道最小值

    def estimate_atmosphere(self, img, dark_ch):
        """估计大气光A"""
        B, H, W = dark_ch.shape
        # 选择暗通道中前0.1%最亮的像素
        num_pixels = int(H * W * self.top_percent)
        flattened_dark = dark_ch.view(B, -1)
        indices = torch.topk(flattened_dark, num_pixels, dim=1)[1]
        # 获取原始图像中对应位置的像素
        atmosphere = []
        for b in range(B):
            selected_pixels = img[b, :, indices[b] // W, indices[b] % W]
            atmosphere.append(torch.max(selected_pixels, dim=1)[0])
        return torch.stack(atmosphere).unsqueeze(-1).unsqueeze(-1)

    def transmission(self, dark_ch):
        """计算透射率图"""
        transmission = 1 - self.omega * dark_ch
        return torch.clamp(transmission, min=self.t0, max=1.0)

    def guided_filter(self, guide, target, radius=40, eps=1e-3):
        """
        guide: 引导图 (B, C, H, W) 通常用灰度图或原图
        target: 待滤波图 (B, 1, H, W) 即透射率图
        """
        # 转换为灰度引导图
        if guide.shape[1] == 3:
            guide_gray = 0.299 * guide[:, 0, :, :] + 0.587 * guide[:, 1, :, :] + 0.114 * guide[:, 2, :, :]
            guide_gray = guide_gray.unsqueeze(1)
        else:
            guide_gray = guide

        # 均值滤波（可用平均池化替代）
        def box_filter(x, r):
            kernel = torch.ones(1, 1, 2 * r + 1, 2 * r + 1).to(x.device) / (2 * r + 1) ** 2
            return nn.functional.conv2d(x, kernel, padding=r, groups=x.shape[1])

        mean_g = box_filter(guide_gray, radius)
        mean_t = box_filter(target, radius)
        mean_gt = box_filter(guide_gray * target, radius)
        mean_gg = box_filter(guide_gray * guide_gray, radius)

        var_g = mean_gg - mean_g * mean_g
        cov_gt = mean_gt - mean_g * mean_t

        a = cov_gt / (var_g + eps)
        b = mean_t - a * mean_g

        mean_a = box_filter(a, radius)
        mean_b = box_filter(b, radius)

        return mean_a * guide_gray + mean_b

    def forward(self, x):
        # 输入形状: (B, C, H, W)，值域[0,1]
        if x.min() < 0:  # 检测到输入是[-1,1]范围
            x = (x + 1) / 2  # 转换到[0,1]
        dark = self.dark_channel(x)
        A = self.estimate_atmosphere(x, dark)
        transmission = self.transmission(dark)
        # 用原图作为引导图，对透射率进行细化
        transmission = self.guided_filter(x, transmission.unsqueeze(1), radius=self.radius)  # (B,1,H,W)
        transmission = transmission.squeeze(1)  # 变回 (B,H,W)
        # 根据物理模型恢复图像
        J = (x - A) / transmission + A
        return torch.clamp(J, 0, 1)

if __name__=="__main__":
    from pyzjr import AverageMeter, BaseDataset
    from torch.utils.data import DataLoader
    from utils import DehazeMetricV1, DehazeDatasetTest

    # gt_path = r"E:\PythonProject\DehazeProject\data\RICE_DATASET\test\GT\21.png"
    # hazy_path = r"E:\PythonProject\DehazeProject\data\RICE_DATASET\test\hazy\21.png"
    PSNR = AverageMeter()
    SSIM = AverageMeter()
    LPIPs = AverageMeter()
    # gt_image = cv2.imread(gt_path)
    # hazy_image = cv2.imread(hazy_path)
    # hazy_image = pyzjr.read_image(hazy_path, 'torch', target_shape=(512, 512)).cuda()
    # target_image = pyzjr.read_image(gt_path, 'torch', target_shape=(512, 512)).cuda()
    data_dir = r'E:\PythonProject\DehazeLab\data\RSHD\thin\test'
    test_dataset = DehazeDatasetTest(data_dir, 512)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             num_workers=2,
                             pin_memory=False)
    network = DCP().cuda()
    for idx, batch in enumerate(test_loader):
        input, target, filename = batch[0].cuda(), batch[1].cuda(), batch[2][0]

        with torch.no_grad():
            output = network(input)
            # [-1, 1] to [0, 1]
            m = DehazeMetricV1(output, target)
            psnr_val, ssim_val, lpips_val = m.get_psnr(), m.get_ssim(), m.get_lpips()

        PSNR.update(psnr_val)
        SSIM.update(ssim_val)
        LPIPs.update(lpips_val)

        print('Test: [{0}]\t'
              'PSNR: {psnr.val:.05f} ({psnr.avg:.05f})\t'
              'SSIM: {ssim.val:.05f} ({ssim.avg:.05f})\t'
              'LPIPS: {lpips.val:.04f} ({lpips.avg:.04f})'
              'filename: {filename}'
              .format(idx + 1, psnr=PSNR, ssim=SSIM, lpips=LPIPs, filename=filename))
