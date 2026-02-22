import os
import cv2
import torch
import numpy as np
import argparse
import pyiqa
from pyzjr.nn import AverageMeter
from pyzjr.visualize.printf import redirect_console
import pyzjr

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"无法读取图像: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return img


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    niqe_metric = pyiqa.create_metric('niqe', device=device)
    brisque_metric = pyiqa.create_metric('brisque', device=device)

    # 使用新的统计类
    NIQE = AverageMeter()
    BRISQUE = AverageMeter()

    # 创建结果目录和日志文件
    result_dir = os.path.join(args.result_dir, args.name)
    time_str = pyzjr.timestr()
    os.makedirs(result_dir, exist_ok=True)

    log_path = os.path.join(result_dir, f'{time_str}_no_ref.log')
    redirect_console(log_path)

    # 显示配置信息
    pyzjr.show_config(args=args)
    print(f'Pred dir: {args.pred_dir}')
    print('-' * 60)

    # 获取图像列表
    img_list = sorted([
        f for f in os.listdir(args.pred_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))
    ])

    print(f"找到 {len(img_list)} 张图像")

    for name in img_list:
        try:
            path = os.path.join(args.pred_dir, name)
            img = load_image(path)

            with torch.no_grad():
                niqe_val = niqe_metric(img.to(device)).item()
                brisque_val = brisque_metric(img.to(device)).item()

            NIQE.update(niqe_val)
            BRISQUE.update(brisque_val)

            print(
                f'Test:{name} '
                f'NIQE: {niqe_val:.4f}({NIQE.avg:.4f})   '
                f'BRISQUE: {brisque_val:.4f}({BRISQUE.avg:.4f})'
            )

        except Exception as e:
            print(f"处理图像 {name} 时出错: {e}")
            continue

    print('\n数值格式:')
    print(f'NIQE: {NIQE.avg:.4f}')
    print(f'BRISQUE: {BRISQUE.avg:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='图像质量评估（无参考指标）')
    parser.add_argument('--pred_dir', default=r'./deresults\PGS2Net_b\2026_02_11_19_02_30/img',
                        help='待评估图像的文件夹路径')
    parser.add_argument('--result_dir', default='./deresults',
                        help='结果保存目录')
    parser.add_argument('--name', default='no_refer',
                        help='实验名称')
    args = parser.parse_args()

    if not os.path.exists(args.pred_dir):
        print(f"错误: 路径不存在 {args.pred_dir}")
        exit(1)

    main(args)

"""
thin
CUHK_CR1 GT
NIQE: 6.5310
BRISQUE: 40.7553
DCP
NIQE: 7.5025
BRISQUE: 47.2949
dehazeformer
NIQE: 5.6312
BRISQUE: 43.8764
PCSformer
NIQE: 5.7380 
BRISQUE: 46.5990
PGS2Net
NIQE: 5.6109
BRISQUE: 43.7359
CUHK_CR1 hazy
NIQE: 6.8968
BRISQUE: 45.9043 

thick
CUHK_CR2 GT
NIQE: 5.9656
BRISQUE: 44.9139
DCP
NIQE: 7.3118
BRISQUE: 50.0770
dehazeformer
NIQE: 5.5827
BRISQUE: 44.3201
PCSformer
NIQE: 5.3885
BRISQUE: 46.3415
PGS2Net
NIQE: 5.5221
BRISQUE: 39.9464

CUHK_CR2 GT
NIQE: 5.9656
BRISQUE: 44.9139
CUHK_CR1 GT
NIQE: 6.5310
BRISQUE: 40.7553
CUHK_CR1 hazy
NIQE: 6.8968
BRISQUE: 45.9043 
CUHK_CR2 hazy
NIQE: 7.1405
BRISQUE: 51.5435


PGS2Net:
SataHaze1K thick
NIQE: 5.9761
BRISQUE: 45.2678

SataHaze1K thin
NIQE: 6.5896
BRISQUE: 48.9525

RSHD thick
NIQE: 5.5221
BRISQUE: 39.9464

RSHD thin
NIQE: 5.6109
BRISQUE: 43.7359
"""



