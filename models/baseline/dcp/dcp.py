# https://blog.csdn.net/m0_62919535/article/details/148291799
import torch
import torch.nn as nn

class DCP(nn.Module):
    def __init__(self, omega=0.95, t0=0.1, top_percent=0.1, radius=40):
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
    from utils import DehazeMetricV1
    import os

    class DehazeDatasetTest(BaseDataset):
        def __init__(
                self,
                root_dir,
                target_shape,
                use_resize=False
        ):
            super(DehazeDatasetTest, self).__init__()
            self.use_resize = use_resize
            self.target_shape = self.to_2tuple(target_shape)
            self.gt = os.path.join(root_dir, 'GT')
            self.hazy = os.path.join(root_dir, 'hazy')
            self.image_name_list = self.SearchFileName(self.gt, ('.png', '.jpg'))

        def __len__(self):
            return len(self.image_name_list)

        def __getitem__(self, item):
            self.disable_cv2_multithreading()
            img_name = self.image_name_list[item]
            source_img = self.read_image(os.path.join(self.hazy, img_name), to_rgb=True, normalize=True) * 2 - 1
            target_img = self.read_image(os.path.join(self.gt, img_name), to_rgb=True, normalize=True) * 2 - 1
            [source_img, target_img] = self.align([source_img, target_img], self.target_shape)
            return self.hwc2chw(source_img), self.hwc2chw(target_img), img_name

    # gt_path = r"E:\PythonProject\DehazeProject\data\RICE_DATASET\test\GT\21.png"
    # hazy_path = r"E:\PythonProject\DehazeProject\data\RICE_DATASET\test\hazy\21.png"
    PSNR = AverageMeter()
    SSIM = AverageMeter()
    LPIPs = AverageMeter()
    # gt_image = cv2.imread(gt_path)
    # hazy_image = cv2.imread(hazy_path)
    # hazy_image = pyzjr.read_image(hazy_path, 'torch', target_shape=(512, 512)).cuda()
    # target_image = pyzjr.read_image(gt_path, 'torch', target_shape=(512, 512)).cuda()
    data_dir = r'E:\PythonProject\DehazeLab\data\RRSHID\thick\test'
    test_dataset = DehazeDatasetTest(data_dir, 256)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             num_workers=2,
                             pin_memory=False)
    network = DCP().cuda()
    for idx, batch in enumerate(test_loader):
        input, target, filename = batch[0].cuda(), batch[1].cuda(), batch[2][0]

        with torch.no_grad():
            output = network(input)
            # pyzjr.imwrite(f"dcp_{filename}", output)
            # [-1, 1] to [0, 1]
            m = DehazeMetricV1(output, target)
            psnr_val, ssim_val, lpips_val = m.get_psnr(), m.get_ssim(), m.get_lpips()

        PSNR.update(psnr_val)
        SSIM.update(ssim_val)
        LPIPs.update(lpips_val)

        print('Test: [{0}]\t'
              'PSNR: {psnr.val:.05f} ({psnr.avg:.05f})\t'
              'SSIM: {ssim.val:.05f} ({ssim.avg:.05f})\t'
              'filename: {filename}'
              .format(idx + 1, psnr=PSNR, ssim=SSIM, filename=filename))
    # import torch.nn.functional as F
    # from pytorch_msssim import ssim
    #
    # def calculate_index(output, target):
    #     # output = output*0.5 + 0.5
    #     # target = target*0.5 + 0.5
    #     psnr = 10 * torch.log10(1/F.mse_loss(output, target)).item()
    #     _, _, H, W = output.size()
    #     down_ratio = max(1, round(min(H, W) / 256))
    #     ssim_val = ssim(F.adaptive_avg_pool2d(output, (int(H / down_ratio), int(W / down_ratio))),
    #                     F.adaptive_avg_pool2d(target, (int(H / down_ratio), int(W / down_ratio))),
    #                     data_range=1, size_average=False).item()
    #     return psnr, ssim_val
    #
    # import pyzjr
    # hazy_path = r"E:\PythonProject\DehazeProject\data\RICE_DATASET\test\hazy\781.png"
    # gt_path   = r"E:\PythonProject\DehazeProject\data\RICE_DATASET\test\GT\781.png"
    # hazy_image = pyzjr.read_image(hazy_path, 'torch', target_shape=(512, 512)).cuda()
    # target_image = pyzjr.read_image(gt_path, 'torch', target_shape=(512, 512)).cuda()
    # dcp = DCP().cuda()
    # out_image = dcp(hazy_image)
    # p, s = calculate_index(out_image, target_image)
    # print(p, s)
    # pyzjr.imwrite("1.png", out_image)


    # 27.51490354537964 0.9374207854270935
    # 25.32953977584839 0.982807993888855  21
