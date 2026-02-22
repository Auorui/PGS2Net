import os
import argparse
import torch
from models import get_dehaze_networks
import pyzjr
from pyzjr.nn import release_gpu_memory, get_lr, AverageMeter, LossHistory, get_optimizer
from pyzjr.visualize.printf import redirect_console
from pyzjr.data import loss_weights_dirs, TrainDataloader, EvalDataloader
from utils import dehaze_criterion, GradualWarmupScheduler, DehazeDataset, DeHazeTrainEpoch


def parse_args(known=False):
    parser = argparse.ArgumentParser(description='Dehazy')
    parser.add_argument('--model', type=str, default='PGS2Net_s',
                        help='train net name')
    parser.add_argument('--resume_training', type=str,
                        default=r'E:\PythonProject\DehazeLab\model_weights\SateHaze1K\Haze1k_thin\PGS2Net_s.pth',
                        help="resume training from last checkpoint")
    parser.add_argument('--dataset_path', type=str,
                        default=r'E:\PythonProject\DehazeLab\data\SateHaze1K\Haze1k_thin',
                        help='dataset path')
    parser.add_argument('--seed', type=int, default=11,
                        help='Random seed number, For example: 11, 42, 3407, 114514, 256')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Training epochs')
    parser.add_argument('--input_shape', default=[256, 256],
                        help='input image shape')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='batch size 2 4 8, if there is a memory overflow, try changing it to 1.'
                             'If there are any further questions, please refer to this article: https://blog.csdn.net/m0_62919535/article/details/132725967')
    parser.add_argument('--log_dir', type=str, default=r'./logs',
                        help='log file path')
    parser.add_argument('--lr', default=2e-4,
                        help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for optimizer')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--optimizer', type=str, default="adamw",
                        help='Optimizer selection, optional adam, adamw, sgd')
    parser.add_argument('--amp', type=bool, default=False,
                        help='Mixed precision training')
    parser.add_argument('--gpu_ids', type=str, default='0',
                        help='GPU IDs to use (e.g., 0,1,2)')
    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    pyzjr.SeedEvery(args.seed)
    loss_log_dir, save_model_dir, timelog_dir = loss_weights_dirs(args.log_dir)
    redirect_console(os.path.join(timelog_dir, 'out.log'))
    pyzjr.show_config(args=args)
    network = get_dehaze_networks(args.model)
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    gpu_ids = [int(id) for id in args.gpu_ids.split(',')]

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_ids[0]}')
    else:
        device = torch.device('cpu')
    print(f"使用设备: {device}")
    network = network.to(device)
    if len(gpu_ids) > 1 and torch.cuda.is_available():
        print(f"使用多卡训练: GPU {gpu_ids}")
        network = torch.nn.DataParallel(network, device_ids=gpu_ids)
    else:
        print(f"使用单卡训练: {device}")
    if args.resume_training is not None:
        print(f"权重 {args.resume_training} 加载到 {args.model}")
        if torch.cuda.is_available():
            checkpoint = torch.load(args.resume_training, map_location=device, weights_only=True)
        else:
            checkpoint = torch.load(args.resume_training, map_location='cpu', weights_only=True)
        if isinstance(network, torch.nn.DataParallel):
            network.module.load_state_dict(checkpoint)
        else:
            network.load_state_dict(checkpoint)
    else:
        print(f"初始训练 {args.model}")

    loss_history = LossHistory(loss_log_dir)

    # criterion = nn.L1Loss()
    criterion = dehaze_criterion()
    optimizer = get_optimizer(
        network,
        optimizer_type=args.optimizer,
        init_lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    train_dataset = DehazeDataset(args.dataset_path, target_shape=args.input_shape,
                                  is_train=True)
    val_dataset = DehazeDataset(args.dataset_path, target_shape=args.input_shape,
                                is_train=False)

    train_loader = TrainDataloader(train_dataset, batch_size=args.batch_size)
    val_loader = EvalDataloader(val_dataset, batch_size=1, num_workers=1)

    warmup_epochs = 3
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                          after_scheduler=scheduler_cosine)

    Defogging = DeHazeTrainEpoch(
        network, args.epochs, optimizer, criterion, use_amp=args.amp
    )
    for epoch in range(args.epochs):
        epoch = epoch + 1
        train_loss = Defogging.train_one_epoch(train_loader, epoch)
        val_loss, psnr = Defogging.evaluate(val_loader, epoch)
        lr_scheduler.step()
        loss_history.append_loss(epoch, train_loss, val_loss)

        print('Epoch:' + str(epoch) + '/' + str(args.epochs))
        print('Total Loss: %.5f || Val Loss: %.5f ' % (train_loss, val_loss))

        pyzjr.SaveModelPth(
            network,
            save_dir=save_model_dir,
            metric=psnr,
            epoch=epoch,
            total_epochs=args.epochs,
            save_period=250
        )
