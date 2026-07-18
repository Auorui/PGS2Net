import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CAP(nn.Module):
    """
    Color Attenuation Prior Dehazing
    PyTorch实现，输入输出为torch张量
    """
    def __init__(
            self,
            beta: float = 1.0,
            guided_filter_radius: int = 60,
            min_filter_radius: int = 15,
            eps: float = 1e-3,
            top_percent: float = 0.001,
            t_min: float = 0.05,
    ):
        """
        Args:
            beta: 大气散射系数，控制去雾强度
            guided_filter_radius: 引导滤波半径
            min_filter_radius: 最小值滤波半径 (对应depth_radius)
            eps: 引导滤波正则化参数
            top_percent: 大气光估计时选取最亮区域的百分比
            t_min: 最小透射率
        """
        super(CAP, self).__init__()
        self.beta = beta
        self.guided_filter_radius = guided_filter_radius
        self.min_filter_radius = min_filter_radius
        self.eps = eps
        self.top_percent = top_percent
        self.t_min = t_min

    def rgb_to_hsv(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        RGB转HSV (模拟OpenCV的cv2.COLOR_RGB2HSV)

        Args:
            rgb: [B, 3, H, W], 值范围 [0, 1]

        Returns:
            hsv: [B, 3, H, W], H:[0,1], S:[0,1], V:[0,1]
        """
        # 分离RGB通道
        r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]

        # 计算最大值、最小值
        max_val, _ = torch.max(rgb, dim=1, keepdim=True)
        min_val, _ = torch.min(rgb, dim=1, keepdim=True)
        delta = max_val - min_val

        # 计算饱和度 S (OpenCV公式)
        s = torch.where(max_val > 0, delta / max_val, torch.zeros_like(max_val))

        # 计算亮度 V
        v = max_val

        # 计算色调 H (OpenCV RGB2HSV公式)
        h = torch.zeros_like(max_val)

        # 当R为最大值时
        mask_r = (r == max_val) & (delta > 0)
        h = torch.where(mask_r, ((g - b) / delta) % 6, h)

        # 当G为最大值时
        mask_g = (g == max_val) & (delta > 0)
        h = torch.where(mask_g, ((b - r) / delta) + 2, h)

        # 当B为最大值时
        mask_b = (b == max_val) & (delta > 0)
        h = torch.where(mask_b, ((r - g) / delta) + 4, h)

        h = h / 6.0  # 归一化到[0,1]
        h = torch.where(delta == 0, torch.zeros_like(h), h)

        return torch.cat([h, s, v], dim=1)

    def guided_filter(self, guide, target, radius=40, eps=1e-3):
        """
        guide: 引导图 (B, C, H, W) 通常用灰度图或原图
        target: 待滤波图 (B, 1, H, W) 即透射率图
        """
        B, C, H, W = guide.shape

        # 转换为灰度引导图
        if guide.shape[1] == 3:
            guide_gray = 0.299 * guide[:, 0:1, :, :] + 0.587 * guide[:, 1:2, :, :] + 0.114 * guide[:, 2:3, :, :]
        else:
            guide_gray = guide

        # 确保target和guide_gray尺寸一致
        if target.shape[2] != H or target.shape[3] != W:
            # 如果尺寸不一致，调整target尺寸
            target = F.interpolate(target, size=(H, W), mode='bilinear', align_corners=False)

        # 均值滤波（可用平均池化替代）
        def box_filter(x, r):
            # 使用更高效的实现
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

    def compute_depth_map(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        计算深度图

        Args:
            rgb: [B, 3, H, W], RGB图像, 值范围 [0, 1]

        Returns:
            depth: [B, 1, H, W] 深度图
        """
        B, C, H, W = rgb.shape

        # 1. RGB转HSV
        hsv = self.rgb_to_hsv(rgb)
        s = hsv[:, 1:2, :, :]  # 饱和度
        v = hsv[:, 2:3, :, :]  # 亮度

        # 2. 计算深度图 (添加高斯噪声模拟随机性)
        sigma = 0.041337
        noise = torch.randn_like(v) * sigma
        depth = 0.121779 + 0.959710 * v - 0.780245 * s + noise

        # 3. 最小值滤波 - 使用更稳健的实现
        r = self.min_filter_radius
        # 使用unfold操作实现最小值滤波，确保尺寸不变
        pad = r
        depth_padded = F.pad(depth, (pad, pad, pad, pad), mode='reflect')
        depth_refine = -F.max_pool2d(-depth_padded, kernel_size=2 * r + 1, stride=1, padding=0)

        # 计算裁剪后的尺寸
        crop_h = depth_refine.shape[2] - 2 * pad
        crop_w = depth_refine.shape[3] - 2 * pad

        # 如果尺寸不匹配，进行裁剪
        if crop_h != H or crop_w != W:
            # 计算裁剪的起始位置
            start_h = (depth_refine.shape[2] - H) // 2
            start_w = (depth_refine.shape[3] - W) // 2
            depth_refine = depth_refine[:, :, start_h:start_h + H, start_w:start_w + W]
        else:
            depth_refine = depth_refine[:, :, pad:pad + H, pad:pad + W]

        # 限制范围
        depth_refine = torch.clamp(depth_refine, 0, 1)

        return depth_refine

    def estimate_atmosphere(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """
        估计大气光

        Args:
            rgb: [B, 3, H, W] RGB图像
            depth: [B, 1, H, W] 深度图

        Returns:
            A: [B, 3, 1, 1] 大气光值
        """
        B, C, H, W = rgb.shape

        # 展平
        depth_flat = depth.view(B, -1)
        rgb_flat = rgb.view(B, C, -1)

        # 计算最亮像素的数量 (0.1%)
        n_bright = max(1, int(self.top_percent * H * W))

        # 获取最亮像素的索引
        _, indices = torch.topk(depth_flat, n_bright, dim=1)

        # 收集候选大气光像素
        Acand = torch.zeros((B, n_bright, C), device=rgb.device)
        for b in range(B):
            Acand[b, :, :] = rgb_flat[b, :, indices[b]].permute(1, 0)

        # 计算每个候选像素的RGB向量范数
        Amag = torch.norm(Acand, dim=2)

        # 选择范数最大的像素
        _, max_idx = torch.max(Amag, dim=1)
        A = torch.zeros((B, C, 1, 1), device=rgb.device)
        for b in range(B):
            A[b, :, 0, 0] = Acand[b, max_idx[b], :]

        return A

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        完整去雾流程

        Args:
            x: [B, 3, H, W] 或 [3, H, W], RGB图像, 值范围 [0, 1]
               如果输入范围是[-1, 1]，会自动转换到[0, 1]

        Returns:
            J: [B, 3, H, W] 或 [3, H, W] 去雾后的RGB图像
        """
        # 检查输入范围并转换
        if x.min() < 0:  # 检测到输入是[-1,1]范围
            x = (x + 1) / 2  # 转换到[0,1]

        # 确保是4维张量
        if x.dim() == 3:
            x = x.unsqueeze(0)
            single_image = True
        else:
            single_image = False

        # 1. 计算深度图
        depth = self.compute_depth_map(x)

        # 2. 引导滤波细化深度图
        depth_refined = self.guided_filter(x, depth)

        # 3. 计算透射率
        transmission = torch.exp(-self.beta * depth_refined)
        transmission = torch.clamp(transmission, min=self.t_min, max=1.0)

        # 4. 估计大气光
        A = self.estimate_atmosphere(x, depth)

        # 5. 图像复原
        J = (x - A) / transmission + A
        J = torch.clamp(J, 0, 1)

        # 恢复原始形状
        if single_image:
            return J.squeeze(0)
        return J



if __name__ == "__main__":
    model = CAP(
        beta=1.0,
        guided_filter_radius=60,
        min_filter_radius=15,
        eps=1e-3
    )
    import cv2

    img = cv2.imread("input.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB
    img_tensor = torch.from_numpy(img).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
    output_np = output.squeeze(0).permute(1, 2, 0).numpy()
    output_np = np.clip(output_np * 255, 0, 255).astype(np.uint8)
    output_np = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite('output_cap.png', output_np)
