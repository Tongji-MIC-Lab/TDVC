import os
import sys
import time
from pathlib import Path

# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
import numpy as np
import torch
import torch.utils.data
import torchvision
import torchvision.datasets.video_utils
from torch import nn

from cfg.default import get_cfg
from dataloader import presets
from dataloader.dataset import Dataset_UCF
from models.model_sr import MODEL
from utils import util

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Loggers(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def evaluate(args, model, data_loader, best_acc):
    model.eval()
    metric_logger = util.MetricLogger(delimiter="  ")
    header = 'Test:'
    hevc, compress, ehc = [], [], []
    with torch.no_grad():
        for video, target, bpp, path, inputs, refs in metric_logger.log_every(data_loader, 400, header):
            video, target = video.to(device, non_blocking=True), target.to(device, non_blocking=True)
            ref_video, raw_video = video[:, :, :args.clip, :, :], video[:, :, args.clip:, :, :]
            with torch.cuda.amp.autocast(enabled=True):
                output, cvideo = model(ref_video, raw_video, inputs, refs, bpp, True)
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
        print(
            ' * currPsnr {cr:.3f}  cPsnr {cp:.3f} hPsnr {hp:.3f} \n'.format(cr=psnr, hp=np.mean(
                np.asarray(hevc)), cp=np.mean(np.asarray(compress)),))


def main(args):
    print(args)
    print("torch version: ", torch.__version__)
    print("torchvision version: ", torchvision.__version__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    model = MODEL(args, args)

    val_data = Dataset_UCF(args.clip, Path(''.join(['/dataset/video_cls/UCF101/', 'UCF-101_fast_yuv_compress_' + str(args.qp) + '_img'])),
                           'validation', presets.VideoClassificationPresetEval())
    data_loader_test = torch.utils.data.DataLoader(val_data, batch_size=1, num_workers=12)


    model_without_ddp = model
    model = torch.nn.DataParallel(model).to(device)

    if args.resume:
        model_without_ddp.load_state_dict(torch.load(args.resume, map_location='cpu')['model'], True)

    evaluate(args, model, data_loader_test, None)
    return


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Video Classification Training')
    parser.add_argument('--model', default='r2plus1d_18', help='model')
    parser.add_argument('--clip', default=8)
    parser.add_argument('--qp', type=int, default=27)
    parser.add_argument('--compress', default=True)
    parser.add_argument('--resume', default='')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    root_path = 'xxx'

    if args.qp == 22:
        args.resume = 'xxx'
    elif args.qp == 27:
        args.resume = 'xxx'
    elif args.qp == 32:
        args.resume = 'xxx'
    else:
        args.resume = 'xxx'

    file_path1 = Path(args.resume).parts
    os.makedirs(os.path.join(root_path, file_path1[-2]), exist_ok=True)
    sys.stdout = Loggers(os.path.join(root_path, file_path1[-2], file_path1[-1]) + '_QP' + str(args.qp) + '.txt')

    main(args)
