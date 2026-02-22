import torch
import os
import numpy as np
import argparse
import pyzjr
from torch.utils.data import DataLoader
from pyzjr.nn import AverageMeter, release_gpu_memory
from pyzjr.visualize.printf import redirect_console
from models.networks import get_dehaze_networks
from utils import DehazeMetricV1, DehazeMetricV2, DehazeDatasetTest

def write_tensor(tensor, filename):
    from PIL import Image
    tensor = tensor.detach().cpu().squeeze(0)
    tensor = tensor.permute(1, 2, 0).numpy()
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    tensor = (tensor * 255).astype(np.uint8)
    image = Image.fromarray(tensor)
    image.save(filename)

def sliding_window_inference(
    model,
    input_tensor,
    window_size=256,
    stride=128
):
    _, C, H, W = input_tensor.shape
    device = input_tensor.device
    flag = None
    if window_size is None or stride is None or (H <= window_size and W <= window_size):
        flag = "Normal inference"
        with torch.no_grad():
            output = model(input_tensor).clamp_(-1, 1)
    else:
        flag = "Sliding inference"
        output = torch.zeros((1, C, H, W), device=device)
        weight = torch.zeros((1, C, H, W), device=device)
        # Hanning window，减少边缘拼接痕迹
        hann_1d = torch.hann_window(window_size, device=device)
        window_weight = hann_1d[:, None] * hann_1d[None, :]
        window_weight = window_weight.unsqueeze(0).unsqueeze(0)  # [1,1,ws,ws]

        for y in range(0, H - window_size + 1, stride):
            for x in range(0, W - window_size + 1, stride):
                patch = input_tensor[:, :, y:y+window_size, x:x+window_size]

                with torch.no_grad():
                    patch_out = model(patch).clamp_(-1, 1)

                output[:, :, y:y+window_size, x:x+window_size] += patch_out * window_weight
                weight[:, :, y:y+window_size, x:x+window_size] += window_weight

        output = output / (weight + 1e-8)  # Prevent division by zero
    return output, flag

def test(args):
    test_dataset = DehazeDatasetTest(
        args.data_dir,
        args.input_shape,
        use_resize=False if args.only_index else args.use_resize
    )
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=2, pin_memory=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = get_dehaze_networks(args.model).to(device)
    if args.model != 'DCP':
        network.load_state_dict(
            torch.load(args.resume_training, map_location='cuda:0')
        )
    network.eval()
    PSNR = AverageMeter()
    SSIM = AverageMeter()
    LPIPS = AverageMeter()
    result_dir = os.path.join(args.result_dir, args.model)
    time_str = pyzjr.timestr()
    save_img_path = os.path.join(result_dir, f'{time_str}/img')
    os.makedirs(save_img_path, exist_ok=True)

    f_result_path = os.path.join(result_dir, f'{time_str}/results.log')
    redirect_console(f_result_path)
    pyzjr.show_config(args=args)

    for idx, batch in enumerate(test_loader):
        input, target, filename = batch[0].to(device), batch[1].to(device), batch[2][0]
        output, flag = sliding_window_inference(
            network,
            input,
            window_size=args.window_size,
            stride=args.stride
        )
        m = DehazeMetricV1(output, target)
        psnr_val, ssim_val, lpips_val = m.get_psnr(), m.get_ssim(), m.get_lpips()

        PSNR.update(psnr_val)
        SSIM.update(ssim_val)
        LPIPS.update(lpips_val)

        print(
            'Test: [{0}]\t'
            'PSNR: {psnr.val:.04f} ({psnr.avg:.04f})  '
            'SSIM: {ssim.val:.04f} ({ssim.avg:.04f})  '
            'LPIPS: {lpips.val:.04f} ({lpips.avg:.04f})  '
            'filename: {filename}  '
            'flag: {flag}'
            .format(idx + 1, psnr=PSNR, ssim=SSIM, lpips=LPIPS, filename=filename, flag=flag)
        )

        if not args.only_index:
            save_path = os.path.join(save_img_path, f'{args.model}_{filename}')
            write_tensor(output, save_path)
            print(f"Save to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='PGS2Net_b', type=str, help='train net name')
    parser.add_argument('--data_dir', default=r'E:\PythonProject\DehazeLab\data\RSHD\thick\test', type=str, help='path to dataset')
    parser.add_argument('--resume_training', default=r'E:\PythonProject\DehazeLab\model_weights\RSHD\thick\PGS2Net_b.pth', type=str,
                        help='path to models saving')
    parser.add_argument('--result_dir', default='./deresults', type=str, help='path to results saving')
    parser.add_argument('--input_shape', default=256, type=int, help='target shape')
    parser.add_argument('--only_index', default=True, type=bool, help='only compute metrics, do not save images')
    parser.add_argument('--use_resize', default=False, type=bool, help='resize input to input_shape, disable sliding window')

    parser.add_argument('--window_size', type=int, default=256,
                        help='patch size for sliding-window inference')
    parser.add_argument('--stride', type=int, default=128,
                        help='stride for sliding-window inference')
    args = parser.parse_args()
    if args.use_resize:
        args.window_size = None
        args.stride = None

    test(args)