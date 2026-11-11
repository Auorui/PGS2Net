import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image


@torch.no_grad()
def sliding_window_inference(
    model,
    input_tensor,
    window_size=256,  # window_size == target_shape
    stride=128,
):
    """
    input_tensor: [1, C, H, W], 范围 [-1, 1]
    return:       [1, C, H, W], 范围 [-1, 1]
    """
    model.eval()
    B, C, H, W = input_tensor.shape
    device = input_tensor.device
    flag = "Sliding inference"
    # 小图直接整图推理
    if H <= window_size and W <= window_size:
        flag = "Normal inference"
        out = model(input_tensor)
        return out.clamp(-1, 1), flag

    # 反射补齐到能被 stride 整除
    pad_h = (stride - (H - window_size) % stride) % stride
    pad_w = (stride - (W - window_size) % stride) % stride
    input_pad = F.pad(input_tensor, (0, pad_w, 0, pad_h), mode='reflect')
    _, _, Hp, Wp = input_pad.shape

    output = torch.zeros((B, C, Hp, Wp), device=device)
    weight = torch.zeros((B, 1, Hp, Wp), device=device)

    # Hanning 窗，减少拼接痕迹
    hann_1d = torch.hann_window(window_size, device=device)
    window_weight = hann_1d[:, None] * hann_1d[None, :]
    window_weight = window_weight.unsqueeze(0).unsqueeze(0)  # [1,1,ws,ws]

    for y in range(0, Hp - window_size + 1, stride):
        for x in range(0, Wp - window_size + 1, stride):
            patch = input_pad[:, :, y:y + window_size, x:x + window_size]
            pred = model(patch).clamp(-1, 1)
            output[:, :, y:y + window_size, x:x + window_size] += pred * window_weight
            weight[:, :, y:y + window_size, x:x + window_size] += window_weight

    output = output / (weight + 1e-8)
    output = output[:, :, :H, :W]
    return output.clamp(-1, 1), flag



def load_image(path):
    """Read image -> tensor [-1, 1], shape [1,3,H,W]"""
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img * 2.0 - 1.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img


def write_tensor(tensor, filename):
    """[1,C,H,W] 范围 [-1,1] -> [H,W,C] uint8 (RGB)。"""
    tensor = tensor.detach().cpu().squeeze(0)
    tensor = (tensor + 1.0) / 2.0
    tensor = tensor.clamp(0, 1)
    img = tensor.permute(1, 2, 0).numpy()
    img = (img * 255).astype('uint8')
    image = Image.fromarray(img)
    image.save(filename)
