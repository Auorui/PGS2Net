import torch

class HaarWavelet:
    @staticmethod
    def wavelet_transform_2d(data):
        """二维 Haar 小波变换，使用矩阵操作实现"""
        B, C, H, W = data.shape
        
        # 确保 H 和 W 是偶数
        if H % 2 != 0:
            data = torch.cat((data, torch.zeros(B, C, 1, W, device=data.device)), dim=2)
            H += 1
        if W % 2 != 0:
            data = torch.cat((data, torch.zeros(B, C, H, 1, device=data.device)), dim=3)
            W += 1

        # 水平变换
        avg_h = (data[:, :, :, 0::2] + data[:, :, :, 1::2]) / 2.0
        detail_h = (data[:, :, :, 0::2] - data[:, :, :, 1::2]) / 2.0

        # 垂直变换
        avg_v = (avg_h[:, :, 0::2, :] + avg_h[:, :, 1::2, :]) / 2.0
        detail_v = (avg_h[:, :, 0::2, :] - avg_h[:, :, 1::2, :]) / 2.0

        avg_d = (detail_h[:, :, 0::2, :] + detail_h[:, :, 1::2, :]) / 2.0
        detail_d = (detail_h[:, :, 0::2, :] - detail_h[:, :, 1::2, :]) / 2.0

        return avg_v, avg_d, detail_v, detail_d, (H, W)

    @staticmethod
    def inverse_wavelet_transform_2d(cA, cH, cV, cD, original_size):
        """二维 Haar 小波逆变换，使用矩阵操作实现"""
        H, W = original_size

        # 垂直逆变换
        avg_h = torch.zeros((cA.shape[0], cA.shape[1], H, cA.shape[3]), device=cA.device)
        avg_h[:, :, 0::2, :] = cA + cV
        avg_h[:, :, 1::2, :] = cA - cV

        detail_h = torch.zeros_like(avg_h)
        detail_h[:, :, 0::2, :] = cH + cD
        detail_h[:, :, 1::2, :] = cH - cD

        # 水平逆变换
        output = torch.zeros((cA.shape[0], cA.shape[1], H, W), device=cA.device)
        output[:, :, :, 0::2] = avg_h + detail_h
        output[:, :, :, 1::2] = avg_h - detail_h

        # 修剪到原始大小
        return output[:, :, :H, :W]
