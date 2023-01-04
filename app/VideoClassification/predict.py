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
from models.model import MODEL
from utils import util


class Loggers(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def evaluate(args, model, criterion, data_loader, device):
    model.eval()
    metric_logger = util.MetricLogger(delimiter="  ")
    header = 'Test:'
    hevcbpp, compressbpp, hevcpsnr, compresspsnr = [], [], [], []

    with torch.no_grad():
        for video, target, bpp, path, inputs, refs in metric_logger.log_every(data_loader, 400, header):
            video, target = video.to(device, non_blocking=True), target.to(device, non_blocking=True)
            ref_video, raw_video = video[:, :, :args.clip, :, :], video[:, :, args.clip:, :, :]
            with torch.cuda.amp.autocast(enabled=args.amp):
                output, bppcompress, psnr_c, psnr_h, recon_video = model(ref_video, raw_video, inputs, refs, args.amp)
            if args.compress:
                bppcompress = np.mean(np.asarray(
                    [bpp.cpu().numpy().transpose(1, 0).tolist()[0]] + torch.stack(bppcompress).cpu().numpy().tolist()))
                compressbpp.append(bppcompress)
            hevcbpp.append(torch.mean(bpp).item())

            acc1, acc5 = util.accuracy(output, target, topk=(1, 5))

            # import torch.nn.functional as F
            # preds_max_prob, preds_index = F.softmax(output, dim=1).max(dim=1)
            # print(preds_index, preds_max_prob)
            # print(target)

            metric_logger.update(loss=criterion(output, target).item())
            metric_logger.meters['acc1'].update(acc1.item(), n=video.shape[0])
            metric_logger.meters['acc5'].update(acc5.item(), n=video.shape[0])
            if args.compress:
                metric_logger.meters['cbpp'].update(bppcompress)
                metric_logger.meters['cpsnr'].update(torch.mean(psnr_c).item())
                compresspsnr.append(torch.mean(psnr_c).item())
            metric_logger.meters['hbpp'].update(torch.mean(bpp).item())
            metric_logger.meters['hpsnr'].update(torch.mean(psnr_h).item())
            hevcpsnr.append(torch.mean(psnr_h).item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    if args.compress:
        print(
            ' * Acc@1 {top1:.3f} Acc@5 {top5:.3f} cBpp {cbpp:.4f} hBpp {hbpp:.4f} cPSNR {cp:.4f} hPSNR {hp:.4f}'.format(
                top1=metric_logger.acc1.global_avg,
                top5=metric_logger.acc5.global_avg,
                hbpp=np.mean(np.asarray(hevcbpp)),
                cbpp=np.mean(np.asarray(compressbpp)),
                cp=np.mean(np.asarray(compresspsnr)),
                hp=np.mean(np.asarray(hevcpsnr))))
    else:
        print(' * Clip Acc@1 {top1.global_avg:.3f} Clip Acc@5 {top5.global_avg:.3f} Bpp {sumbpp:.4f} hPSNR {hp:.4f}'
              .format(top1=metric_logger.acc1, top5=metric_logger.acc5, sumbpp=np.mean(np.asarray(hevcbpp)),
                      hp=np.mean(np.asarray(hevcpsnr))))


def main(args):
    print(args)
    print("torch version: ", torch.__version__)
    print("torchvision version: ", torchvision.__version__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    if args.vcm:
        if args.qp == 22:
            args.compress_path = 'xxx'
        elif args.qp == 27:
            args.compress_path = 'xxx'
        elif args.qp == 32:
            args.compress_path = 'xxx'
        else:
            args.compress_path = 'xxx'
    model = MODEL(args, args)
    model.videocls.fc = nn.Linear(512, 101)

    val_data = Dataset_UCF(args.clip, Path(''.join(
        ['/dataset/video_cls/UCF101/', 'UCF-101_fast_yuv_compress_' + str(args.qp) + '_img'])),
                           'validation', presets.VideoClassificationPresetEval())
    data_loader_test = torch.utils.data.DataLoader(val_data, batch_size=2, num_workers=12)

    criterion = nn.CrossEntropyLoss()

    model_without_ddp = model
    model = torch.nn.DataParallel(model).to(device)

    if args.resume:
        if args.vcm:
            dicts = torch.load(args.resume, map_location='cpu')['model']
            new_dict = {}
            for k, v in dicts.items():
                if 'videocls' in k:
                    new_dict.setdefault(k, v)
            model_without_ddp.load_state_dict(new_dict, False)
        else:
            model_without_ddp.load_state_dict(torch.load(args.resume, map_location='cpu')['model'], True)

    evaluate(args, model, criterion, data_loader_test, device)
    return


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Video Classification Training')
    parser.add_argument('--model', default='r2plus1d_18', help='model')
    parser.add_argument('--clip', default=8)
    parser.add_argument('--qp', type=int, default=22)
    parser.add_argument('--compress', action='store_true')
    parser.add_argument('--compress_path', default='')
    parser.add_argument('--amp', default=False)
    parser.add_argument('--vcm', default=True)
    parser.add_argument('--resume', default='')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    root_path = 'xxx'

    if args.compress:
        if args.qp == 22:
            args.resume = 'xxx'
        elif args.qp == 27:
            args.resume = 'xxx'
        elif args.qp == 32:
            args.resume = 'xxx'
        else:
            args.resume = 'xxx'
    else:
        args.resume = 'xxx'

    if args.vcm:
        args.resume = 'xxx'
    file_path1 = Path(args.resume).parts
    os.makedirs(os.path.join(root_path, file_path1[-2]), exist_ok=True)
    sys.stdout = Loggers(os.path.join(root_path, file_path1[-2], file_path1[-1]) + '_QP' + str(args.qp) + '.txt')

    main(args)
