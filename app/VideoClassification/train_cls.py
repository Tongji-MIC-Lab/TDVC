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
from torch import nn
from torch.utils.data import DataLoader

from cfg.default import get_cfg
from dataloader import presets
from dataloader.dataset import Dataset_UCF
from models.model import MODEL
from utils import util
from utils.mutil_task_opt import AutomaticWeightedLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

awl = AutomaticWeightedLoss().to(device)
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


def train_one_epoch(model, criterion, data_loader, epoch, optimizers):
    if opt.compress:
        optimizer, optimizer_mv, optimizer_res = optimizers
    else:
        optimizer = optimizers
    model.train()
    metric_logger = util.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', util.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    for video, target, bpp, path, inputs, refs in metric_logger.log_every(data_loader, 200, header):
        video, target = video.to(device), target.to(device)
        ref_video, raw_video = video[:, :, :opt.clip, :, :], video[:, :, opt.clip:, :, :]

        if opt.compress:
            with torch.cuda.amp.autocast(enabled=opt.amp):
                output, mse, bpp, mv_loss, res_loss, recon_video = model(ref_video, raw_video, inputs, refs, opt.amp)
                bpp, mv_loss, res_loss, mse_loss = torch.mean(bpp), torch.mean(mv_loss), torch.mean(res_loss), torch.mean(mse)

                closs = criterion(output, target)
                rd_loss = opt.lambda_ * mse_loss
                loss, s1, s2 = awl(rd_loss, closs)
                loss = bpp + loss

            optimizer.zero_grad()
            optimizer_mv.zero_grad()
            optimizer_res.zero_grad()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            scaler.step(optimizer)
            scaler.update()

            mv_loss.backward()
            res_loss.backward()
            optimizer_mv.step()
            optimizer_res.step()

            metric_logger.meters['bpp'].update(bpp.item())
            metric_logger.meters['rd_loss'].update(rd_loss.item())
            metric_logger.meters['cls_loss'].update(closs.item())

            metric_logger.meters['w1'].update(s1)
            metric_logger.meters['w2'].update(s2)
        else:
            output = model(ref_video, raw_video, inputs, refs, False)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()

        acc1, acc5 = util.accuracy(output, target, topk=(1, 5))
        metric_logger.update(loss=loss.item(), lr=np.array([x['lr'] for x in optimizer.param_groups])[0])
        metric_logger.meters['acc1'].update(acc1.item(), n=video.shape[0])
        metric_logger.meters['acc5'].update(acc5.item(), n=video.shape[0])


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

def evaluate(args, model, criterion, data_loader, best_acc, cor):
    model.eval()
    metric_logger = util.MetricLogger(delimiter="  ")
    header = 'Test:'
    hevcbpp, compressbpp, hevcpsnr, compresspsnr = [], [], [], []
    with torch.no_grad():
        for video, target, bpp, path, inputs, refs in metric_logger.log_every(data_loader, args.print_freq * 2, header):
            video, target = video.to(device, non_blocking=True), target.to(device, non_blocking=True)
            ref_video, raw_video = video[:, :, :opt.clip, :, :], video[:, :, opt.clip:, :, :]
            with torch.cuda.amp.autocast(enabled=False):
                output, bppcompress, psnr_c, psnr_h, recon_video = model(ref_video, raw_video, inputs, refs, False)

            """
            mse_loss1 = torch.nn.MSELoss()(recon_video, raw_video)
            mse_loss2 = torch.nn.MSELoss()(ref_video, raw_video)
            psnr_c = 10 * (torch.log(1 * 1 / mse_loss1) / np.log(10)).cpu().numpy() if mse_loss1 > 0 else 100
            psnr_h = 10 * (torch.log(1 * 1 / mse_loss2) / np.log(10)).cpu().numpy() if mse_loss2 > 0 else 100
            """

            if opt.compress:
                bppcompress = np.mean(np.asarray(
                    [bpp.cpu().numpy().transpose(1, 0).tolist()[0]] + torch.stack(bppcompress).cpu().numpy().tolist()))
                compressbpp.append(bppcompress)
            hevcbpp.append(torch.mean(bpp).item())

            acc1, acc5 = util.accuracy(output, target, topk=(1, 5))
            metric_logger.update(loss=criterion(output, target).item())
            metric_logger.meters['acc1'].update(acc1.item(), n=video.shape[0])
            metric_logger.meters['acc5'].update(acc5.item(), n=video.shape[0])
            if opt.compress:
                metric_logger.meters['cbpp'].update(bppcompress)
                metric_logger.meters['cpsnr'].update(torch.mean(psnr_c).item())
                compresspsnr.append(torch.mean(psnr_c).item())
            metric_logger.meters['hbpp'].update(torch.mean(bpp).item())
            metric_logger.meters['hpsnr'].update(torch.mean(psnr_h).item())
            hevcpsnr.append(torch.mean(psnr_h).item())

            # gather the stats from all processes
        metric_logger.synchronize_between_processes()

        corbpp, corcpsnr, corhpsnr = cor
        if opt.compress:
            if metric_logger.acc1.global_avg > best_acc:
                best_acc = metric_logger.acc1.global_avg
                corbpp = np.mean(np.asarray(compressbpp))
                corcpsnr = np.mean(np.asarray(compresspsnr))
                corhpsnr = np.mean(np.asarray(hevcpsnr))
            print(
                ' * Acc@1 {top1:.3f} Acc@5 {top5:.3f} cBpp {cbpp:.4f} hBpp {hbpp:.4f} cPSNR {cp:.4f} hPSNR {hp:.4f} \n'
                ' * Best Acc@1 {bestacc:.3f} bpp {bpp:.4f} cPSNR {bcp:.4f} hPSNR {bhp:.4f}'.format(
                    top1=metric_logger.acc1.global_avg,
                    top5=metric_logger.acc5.global_avg,
                    hbpp=np.mean(np.asarray(hevcbpp)),
                    cbpp=np.mean(np.asarray(compressbpp)),
                    cp=np.mean(np.asarray(compresspsnr)),
                    hp=np.mean(np.asarray(hevcpsnr)),
                    bestacc=best_acc, bpp=corbpp,
                    bcp=corcpsnr, bhp=corhpsnr))
        else:
            if metric_logger.acc1.global_avg > best_acc:
                best_acc = metric_logger.acc1.global_avg
                corbpp = np.mean(np.asarray(hevcbpp))
                corhpsnr = np.mean(np.asarray(hevcpsnr))
            print(' * Acc@1 {top1:.3f} Acc@5 {top5:.3f} hBpp {hbpp:.4f} hPSNR {hp:.4f} \n '
                  '* Best Acc {bestacc:.3f} bpp {bpp:.4f} hPSNR {bhp:.4f}'.format(top1=metric_logger.acc1.global_avg,
                                                                                  top5=metric_logger.acc5.global_avg,
                                                                                  hbpp=np.mean(np.asarray(hevcbpp)),
                                                                                  hp=np.mean(np.asarray(hevcpsnr)),
                                                                                  bestacc=best_acc, bpp=corbpp,
                                                                                  bhp=corhpsnr))
        return best_acc, (corbpp, corcpsnr, corhpsnr)


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
    model.videocls.fc = nn.Linear(512, 101)

    criterion = nn.CrossEntropyLoss()

    if opt.compress:
        optimizers = util.configure_optimizers(awl, opt.lr, model)
        print(optimizers[0], '\n')
        print(optimizers[1], '\n')
        print(optimizers[2], '\n')
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
        print(optimizer, '\n')

    model_without_ddp = model
    model = torch.nn.DataParallel(model).to(device)

    if opt.resume:
        # model_without_ddp.load_state_dict(torch.load(opt.resume, map_location='cpu')['model'], strict=False)
        dicts = torch.load(opt.resume, map_location='cpu')['model']
        new_dict = {}
        for k, v in dicts.items():
            if 'videocls' in k:
                new_dict.setdefault(k, v)
        model_without_ddp.load_state_dict(new_dict, False)

    train_data = Dataset_UCF(opt.clip, Path(''.join([opt.dataset_path, 'UCF-101_yuv_compress_' + str(opt.qp) + '_img'])),
                             'training', presets.VideoClassificationPresetTrain())
    val_data = Dataset_UCF(opt.clip, Path(''.join([opt.dataset_path, 'UCF-101_yuv_compress_' + str(opt.qp) + '_img'])),
                           'validation', presets.VideoClassificationPresetEval())

    data_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
    data_loader_test = DataLoader(val_data, batch_size=2, num_workers=opt.workers)

    best_acc, corbpp, corcpsnr, corhpsnr = 0, 0, 0, 0
    cor = (corbpp, corcpsnr, corhpsnr)
    # best_acc, cor = evaluate(args, model, criterion, data_loader_test, best_acc, cor)
    for epoch in range(args.start_epoch, args.epochs):
        if opt.compress:
            train_one_epoch(model, criterion, data_loader, epoch, optimizers)
        else:
            train_one_epoch(model, criterion, data_loader, epoch, optimizer)

        if opt.output_dir:
            checkpoint = {'model': model_without_ddp.state_dict()}
            util.save_on_master(checkpoint, os.path.join(opt.output_dir, 'model_{}.pth'.format(epoch)))

        torch.cuda.empty_cache()

        if opt.compress:
            best_acc, cor = evaluate(args, model, criterion, data_loader_test, best_acc, cor)
        else:
            best_acc, cor = evaluate(args, model, criterion, data_loader_test, best_acc, cor)

        torch.cuda.empty_cache()


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Video Classification Training')
    parser.add_argument('--cfg', default='cfg/compress.yaml')
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
