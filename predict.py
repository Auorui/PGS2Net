import os
import torch
import argparse
from tqdm import tqdm
import pyzjr
from models.networks import get_dehaze_networks
from utils import sliding_window_inference, load_image, write_tensor


def predict(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_dehaze_networks(args.model).to(device)
    if args.model not in ('DCP', 'CAP'):
        model.load_state_dict(torch.load(args.weight, map_location=device))
    model.eval()

    result_dir = os.path.join(args.result_dir, args.model)
    time_str = pyzjr.timestr()
    save_img_path = os.path.join(result_dir, f'{time_str}/img')
    os.makedirs(save_img_path, exist_ok=True)

    img_list = pyzjr.get_image_path(args.input_dir)
    pyzjr.show_config(args=args)

    for name in tqdm(img_list):
        img = load_image(os.path.join(args.input_dir, name)).to(device)
        output, _ = sliding_window_inference(
            model, img,
            window_size=args.window_size,
            stride=args.stride,
        )
        write_tensor(output, os.path.join(save_img_path, name))
    print(f'Inference done. Results saved to: {save_img_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='PGS2Net_b')
    parser.add_argument('--weight', type=str, default=r'model_weights\RSHD\thick\PGS2Net_b.pth')
    parser.add_argument('--input_dir', type=str,
                        default=r'E:\PythonProject\DehazeLab\data\RSHD\thick\test\hazy')
    parser.add_argument('--result_dir', type=str, default=r'./work_dir')
    parser.add_argument('--window_size', type=int, default=256)
    parser.add_argument('--stride', type=int, default=128)
    args = parser.parse_args()
    predict(args)
