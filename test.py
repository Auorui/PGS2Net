import os
import argparse
import torch
import pyzjr
from torch.utils.data import DataLoader
from pyzjr.nn import AverageMeter
from pyzjr.utils import redirect_console

from models.networks import get_dehaze_networks
from utils import DehazeMetricV1, DehazeDatasetTest, sliding_window_inference, write_tensor


def test(args):
    test_dataset = DehazeDatasetTest(args.data_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=2, pin_memory=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    network = get_dehaze_networks(args.model).to(device)
    if args.model not in ('DCP', 'CAP'):
        network.load_state_dict(torch.load(args.weight, map_location=device))
    network.eval()

    PSNR, SSIM, LPIPS = AverageMeter(), AverageMeter(), AverageMeter()

    if not args.only_index:
        result_dir = os.path.join(args.result_dir, args.model)
        time_str = pyzjr.timestr()
        save_img_path = os.path.join(result_dir, f'{time_str}/img')
        os.makedirs(save_img_path, exist_ok=True)
        f_result_path = os.path.join(result_dir, f'{time_str}/results.log')
        redirect_console(f_result_path)

    pyzjr.show_config(args=args)

    for idx, batch in enumerate(test_loader):
        input_img = batch[0].to(device)
        target = batch[1].to(device)
        filename = batch[2][0]

        output, flag = sliding_window_inference(
            network, input_img,
            window_size=args.window_size, stride=args.stride,
        )

        m = DehazeMetricV1(output, target)
        psnr_val, ssim_val, lpips_val = m.get_psnr(), m.get_ssim(), m.get_lpips()
        if torch.isnan(torch.tensor(psnr_val)):
            print(f"Skipping {filename} due to NaN PSNR")
            continue

        PSNR.update(psnr_val)
        SSIM.update(ssim_val)
        LPIPS.update(lpips_val)

        print(
            'Test: [{0}]\t'
            'PSNR: {psnr.val:.04f} ({psnr.avg:.04f})  '
            'SSIM: {ssim.val:.04f} ({ssim.avg:.04f})  '
            'LPIPS: {lpips.val:.04f} ({lpips.avg:.04f})  '
            'filename: {filename}  '
            'flag: {flag}'.format(
                idx + 1, psnr=PSNR, ssim=SSIM, lpips=LPIPS, filename=filename, flag=flag
            )
        )

        if not args.only_index:
            if not os.path.splitext(filename)[1]:
                filename = filename + '.png'
            save_path = os.path.join(save_img_path, filename)
            write_tensor(output, save_path)
            print(f"Save to {save_path}")

    print(f"\nPSNR & SSIM & LPIPS: {PSNR.avg:.4} & {SSIM.avg:.4} & {LPIPS.avg:.4}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='PGS2Net_s', type=str)
    parser.add_argument('--data_dir', default=r'E:\PythonProject\DehazeLab\data\RSHD\thick\test', type=str)
    parser.add_argument('--weight',
                        default=r'E:\PythonProject\DehazeLab\model_weights\RSHD\thick\PGS2Net_s.pth', type=str)
    parser.add_argument('--result_dir', default=r'./results', type=str)
    parser.add_argument('--only_index', default=True, type=bool)
    parser.add_argument('--window_size', type=int, default=256)
    parser.add_argument('--stride', type=int, default=128)
    args = parser.parse_args()
    test(args)
