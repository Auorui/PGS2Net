import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from tqdm import tqdm
import pyzjr
from models.networks import get_dehaze_networks


def load_image(path):
    """Read image -> tensor [-1, 1], shape [1,3,H,W]"""
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img * 2.0 - 1.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img


def save_image(tensor, path):
    """Tensor [-1,1] -> uint8 image"""
    tensor = tensor.detach().cpu().squeeze(0)
    tensor = (tensor + 1.0) / 2.0
    tensor = tensor.clamp(0, 1)
    img = tensor.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def sliding_window_inference(
    model,
    input_tensor,
    window_size=256,
    stride=128
):
    """
    input_tensor: [1, C, H, W], range [-1, 1]
    return: [1, C, H, W]
    """
    B, C, H, W = input_tensor.shape
    device = input_tensor.device

    pad_h = (stride - (H - window_size) % stride) % stride
    pad_w = (stride - (W - window_size) % stride) % stride

    input_pad = F.pad(input_tensor, (0, pad_w, 0, pad_h), mode='reflect')
    _, _, Hp, Wp = input_pad.shape

    output = torch.zeros((B, C, Hp, Wp), device=device)
    weight = torch.zeros((B, 1, Hp, Wp), device=device)

    # Hanning window
    hann_1d = torch.hann_window(window_size, device=device)
    window_weight = hann_1d[:, None] * hann_1d[None, :]
    window_weight = window_weight.unsqueeze(0).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        for y in range(0, Hp - window_size + 1, stride):
            for x in range(0, Wp - window_size + 1, stride):
                patch = input_pad[:, :, y:y+window_size, x:x+window_size]
                pred = model(patch)
                output[:, :, y:y+window_size, x:x+window_size] += pred * window_weight
                weight[:, :, y:y+window_size, x:x+window_size] += window_weight

    output = output / (weight + 1e-8)
    output = output[:, :, :H, :W]

    return output.clamp(-1, 1)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_dehaze_networks(args.model).to(device)
    if args.model != 'DCP':
        model.load_state_dict(torch.load(args.weight, map_location=device))
    model.eval()

    result_dir = os.path.join(args.result_dir, args.model)
    time_str = pyzjr.timestr()
    save_img_path = os.path.join(result_dir, f'{time_str}/img')
    os.makedirs(save_img_path, exist_ok=True)

    img_list = [
        f for f in os.listdir(args.input_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))
    ]

    for name in tqdm(img_list):
        img_path = os.path.join(args.input_dir, name)

        img = load_image(img_path).to(device)
        output = sliding_window_inference(
            model,
            img,
            window_size=args.window_size,
            stride=args.stride
        )
        save_path = os.path.join(save_img_path, name)
        save_image(output, save_path)

    print(f'Inference done. Results saved to: {save_img_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='PGS2Net_b')
    parser.add_argument('--weight', type=str, default=r'model_weights\RSHD\thin\PGS2Net_b.pth')
    parser.add_argument('--input_dir', type=str, default=r'data\CCUHK\CUHK_CR1\test\hazy',
                        help='folder of real hazy images')
    parser.add_argument('--result_dir', type=str, default=r'./deresults',
                        help='folder to save dehazed images')

    parser.add_argument('--window_size', type=int, default=256)
    parser.add_argument('--stride', type=int, default=128)

    args = parser.parse_args()
    main(args)
