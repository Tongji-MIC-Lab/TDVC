import glob
import os
import random
import shutil
import sys
from pathlib import Path

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from cfg.default import get_cfg
from dataloader import presets
from dataloader.dataset import Dataset_UCF
from models.model_sr import MODEL
from utils import util
from utils.mutil_task_opt import CharbonnierLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

scaler = torch.cuda.amp.GradScaler()


class Loggers(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def train_one_epoch(model, criterion, data_loader, epoch, optimizer):
    model.train()
    metric_logger = util.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', util.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    for video, target, bpp, path, inputs, refs in metric_logger.log_every(data_loader, 200, header):
        video, target = video.to(device), target.to(device)
        ref_video, raw_video = video[:, :, :opt.clip, :, :], video[:, :, opt.clip:, :, :]

        with torch.cuda.amp.autocast(enabled=opt.amp):
            output, cvideo = model(ref_video, raw_video, inputs, refs, opt.amp)
            mse1 = torch.nn.MSELoss()(ref_video, raw_video)
            mse2 = torch.nn.MSELoss()(cvideo, raw_video)
            mse3 = torch.nn.MSELoss()(output, raw_video)
            psnr1 = 10 * (torch.log(1 * 1 / mse1) / np.log(10)).cpu().detach().numpy() if mse1 > 0 else 100
            psnr2 = 10 * (torch.log(1 * 1 / mse2) / np.log(10)).cpu().detach().numpy() if mse2 > 0 else 100
            psnr3 = 10 * (torch.log(1 * 1 / mse3) / np.log(10)).cpu().detach().numpy() if mse3 > 0 else 100

            loss = criterion(output, raw_video)
        optimizer.zero_grad()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        scaler.step(optimizer)
        scaler.update()

        metric_logger.meters['ref'].update(psnr1.item())
        metric_logger.meters['compress'].update(psnr2.item())
        metric_logger.meters['ehc'].update(psnr3.item())
        metric_logger.update(loss=loss.item(), lr=np.array([x['lr'] for x in optimizer.param_groups])[0])


import math


def cpsnr2(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def evaluate(args, model, data_loader, best_acc):
    model.eval()
    metric_logger = util.MetricLogger(delimiter="  ")
    header = 'Test:'
    hevc, compress, ehc = [], [], []
    with torch.no_grad():
        for video, target, bpp, path, inputs, refs in metric_logger.log_every(data_loader, args.print_freq * 2, header):
            video, target = video.to(device, non_blocking=True), target.to(device, non_blocking=True)
            ref_video, raw_video = video[:, :, :opt.clip, :, :], video[:, :, opt.clip:, :, :]
            with torch.cuda.amp.autocast(enabled=opt.amp):
                output, cvideo = model(ref_video, raw_video, inputs, refs, opt.amp)
                mse1 = torch.nn.MSELoss()(ref_video, raw_video)
                mse2 = torch.nn.MSELoss()(cvideo, raw_video)
                mse3 = torch.nn.MSELoss()(output, raw_video)
                psnr1 = 10 * (torch.log(1 * 1 / mse1) / np.log(10)).cpu().detach().numpy() if mse1 > 0 else 100
                psnr2 = 10 * (torch.log(1 * 1 / mse2) / np.log(10)).cpu().detach().numpy() if mse2 > 0 else 100
                psnr3 = 10 * (torch.log(1 * 1 / mse3) / np.log(10)).cpu().detach().numpy() if mse3 > 0 else 100

                hevc.append(psnr1)
                compress.append(psnr2)
                ehc.append(psnr3)

            metric_logger.meters['ref'].update(psnr1)
            metric_logger.meters['compress'].update(psnr2)
            metric_logger.meters['ehc'].update(psnr3)
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()

        psnr = np.mean(np.asarray(ehc))
        if psnr > best_acc:
            best_acc = psnr

        print(
            ' * currPsnr {cr:.3f}  hPsnr {hp:.3f} cPsnr {cp:.3f} \n * bestPsnr {best:.3f} '.format(cr=psnr, hp=np.mean(
                np.asarray(hevc)), cp=np.mean(np.asarray(compress)), best=best_acc))

        return best_acc


def main():
    print('\nConfig:')
    print(opt)
    print(args)
    print("torch version: ", torch.__version__)
    print("torchvision version: ", torchvision.__version__)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    random.seed(1111)
    np.random.seed(1111)
    torch.manual_seed(1111)
    torch.cuda.manual_seed_all(1111)

    model = MODEL(args, opt)

    criterion = CharbonnierLoss()
    parameters = {n for n, p in model.named_parameters()}
    params_dict = dict(model.named_parameters())
    optimizers = torch.optim.Adam((params_dict[n] for n in sorted(parameters) if 'ehc_model' in n), lr=0.0001)
    print(optimizers, '\n')

    model_without_ddp = model
    model = torch.nn.DataParallel(model).to(device)

    if opt.resume:
        # model_without_ddp.load_state_dict(torch.load(opt.resume, map_location='cpu')['model'], strict=False)
        dicts = torch.load(opt.resume, map_location='cpu')['model']
        new_dict = {}
        for k, v in dicts.items():
            if 'videocompress' in k:
                new_dict.setdefault(k, v)
        model_without_ddp.load_state_dict(new_dict, False)
    dataset_path = '/dataset/video_cls/UCF101/'

    train_data = Dataset_UCF(opt.clip, Path(''.join([dataset_path, 'UCF-101_fast_yuv_compress_' + str(opt.qp) + '_img'])),
                             'training', presets.VideoClassificationPresetTrainEHC())
    val_data = Dataset_UCF(opt.clip, Path(''.join([dataset_path, 'UCF-101_fast_yuv_compress_' + str(opt.qp) + '_img'])),
                           'validation', presets.VideoClassificationPresetEval())

    data_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
    data_loader_test = DataLoader(val_data, batch_size=2, num_workers=opt.workers)

    best_acc = 0
    # best_acc = evaluate(args, model, data_loader_test, best_acc)
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, criterion, data_loader, epoch, optimizers)
        if opt.output_dir:
            checkpoint = {'model': model_without_ddp.state_dict()}
            util.save_on_master(checkpoint, os.path.join(opt.output_dir, 'model_{}.pth'.format(epoch)))
        torch.cuda.empty_cache()
        best_acc = evaluate(args, model, data_loader_test, best_acc)
        torch.cuda.empty_cache()


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Video Classification Training')
    parser.add_argument('--cfg', default='cfg/ehc.yaml')
    parser.add_argument('--model', default='r2plus1d_18', help='model')
    parser.add_argument('--epochs', default=50, type=int, metavar='N')
    parser.add_argument('--print-freq', default=200, type=int, help='print frequency')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')
    args = parser.parse_args()

    opt = get_cfg()
    opt.merge_from_file(args.cfg)
    return args, opt


if __name__ == "__main__":
    args, opt = parse_args()
    list(map(shutil.rmtree, glob.glob(opt.output_dir)))
    os.makedirs(opt.output_dir, exist_ok=True)
    sys.stdout = Loggers(opt.output_dir + '/log.txt')
    os.system('cp train_cls.py ' + opt.output_dir)
    main()
