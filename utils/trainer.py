import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from pyzjr.nn import release_gpu_memory, get_lr, AverageMeter, LossHistory, get_optimizer
from utils.metric import DehazeMetricV1, DehazeMetricV2

class DeHazeTrainEpoch(object):
    def __init__(self,
                 model,
                 total_epoch,
                 optimizer,
                 # lr_scheduler,
                 loss_function,
                 use_amp=False,
                 device=torch.device("cuda:0")):
        super(DeHazeTrainEpoch, self).__init__()
        self.device = device
        self.model = model.to(device)
        self.loss_function = loss_function
        self.optimizer = optimizer
        # self.lr_scheduler = lr_scheduler
        self.total_epoch = total_epoch
        release_gpu_memory()
        self.scaler = None
        if use_amp:
            self.scaler = GradScaler()

    def train_one_epoch(self, train_loader, epoch):
        self.model.train()
        losses = AverageMeter()
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{self.total_epoch}', postfix=dict,
                  mininterval=0.3) as pbar:
            for batch in train_loader:
                source_img, target_img = batch[0].to(self.device).float(), \
                                         batch[1].to(self.device).float()
                with autocast(enabled=self.scaler is not None):
                    outputs = self.model(source_img)
                self.optimizer.zero_grad()
                if self.scaler is not None:
                    loss = torch.nan_to_num(self.loss_function(outputs, target_img, source_img))
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss = self.loss_function(outputs, target_img, source_img)
                    loss.backward()
                    self.optimizer.step()
                # self.lr_scheduler.step()  # Placed after the end of each round
                losses.update(loss.item())
                pbar.set_postfix(**{'train_loss': losses.avg,
                                    'lr': get_lr(self.optimizer)})
                pbar.update(1)
        return losses.avg

    def evaluate(self, val_loader, epoch):
        self.model.eval()
        PSNR = AverageMeter()
        SSIM = AverageMeter()
        losses = AverageMeter()
        with tqdm(total=len(val_loader), desc=f'Epoch {epoch}/{self.total_epoch}', postfix=dict,
                  mininterval=0.3) as pbar:
            for batch in val_loader:
                source_img, target_img = batch[0].to(self.device), batch[1].to(self.device)
                with torch.no_grad():
                    outputs = self.model(source_img).clamp_(-1, 1)
                    loss = self.loss_function(outputs, target_img, source_img)
                    metric_v1 = DehazeMetricV1(outputs, target_img)
                    psnr_val, ssim_val = metric_v1.get_psnr(), metric_v1.get_ssim()
                PSNR.update(psnr_val, source_img.size(0))
                SSIM.update(ssim_val, source_img.size(0))
                losses.update(loss.item())
                pbar.set_postfix(**{'psnr': PSNR.avg})
                pbar.update(1)
            print(f"{epoch} - psnr: {PSNR.avg}, ssim: {SSIM.avg}")
        return losses.avg, PSNR.avg