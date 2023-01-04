import argparse
import datetime
import glob
import importlib
import json
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
from main.dataloader.dataset import DataSet, HEVCDataSet
from main.model.ms_ssim_torch import ms_ssim
from main.utils.utils import configure_optimizers, crop, pad

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def Var(x):
    return Variable(x.to(device))


def validation(val_loader, net):
    psnrs, bpps, ssims, mvs, ress, mses = [], [], [], [], [], []
    for batch_idx, input in enumerate(val_loader):
        input_images, ref_image, ref_bpp, ref_psnr, ref_msssim = input[0], input[1], input[2], input[3], input[4]

        for b in range(input_images.size(0)):
            bpps.append(float(ref_bpp[b].detach().numpy()))
            psnrs.append(float(ref_psnr[b].detach().numpy()))
            ssims.append(float(ref_msssim[b].detach().numpy()))

        ref_imglist = [Var(pad(ref_image, 64))]
        for i in range(input_images.size()[1]):
            inputframe = Var(pad(input_images[:, i, :, :, :], 64))

            if len(ref_imglist) == 1:
                ref_frames_ = torch.stack([ref_imglist[0], ref_imglist[-1], ref_imglist[-1], ref_imglist[-1]])
            elif len(ref_imglist) == 2:
                ref_frames_ = torch.stack([ref_imglist[0], ref_imglist[-2], ref_imglist[-1], ref_imglist[-1]])
            else:
                ref_frames_ = torch.stack([ref_imglist[0], ref_imglist[-3], ref_imglist[-2], ref_imglist[-1]])

            refframes_ = Var(ref_frames_.transpose(1, 0).view(-1, 4, 3, inputframe.size(2), inputframe.size(3)))

            with torch.cuda.amp.autocast(enabled=enable_amp):
                recon_image, bpp_res, bpp_mv = net(inputframe, refframes_, enable_amp)
            ref_imglist.append(recon_image)
            recon_image = crop(recon_image, (ref_image.size(2), ref_image.size(3)))
            inputframe = crop(inputframe, (ref_image.size(2), ref_image.size(3)))

            for b in range(recon_image.size(0)):
                mse_loss = torch.nn.MSELoss()(recon_image[b], inputframe[b])
                psnr = float(10 * (torch.log(1. / mse_loss) / np.log(10)).cpu().detach().numpy())
                psnrs.append(psnr)
                bpps.append(float((bpp_res + bpp_mv)[b].cpu().detach().numpy()))
                ssims.append(float(ms_ssim(recon_image[b].float().unsqueeze(0), inputframe[b].unsqueeze(0),
                                           data_range=1.0).cpu().detach().numpy()))
                mvs.append(float(bpp_mv[b]))
                ress.append(float(bpp_res[b]))
                mses.append(float(mse_loss))

    return np.mean(bpps), np.mean(mvs), np.mean(ress), np.mean(psnrs), np.mean(ssims), np.mean(mses)


def train():
    """ DataSet """
    train_dataset = DataSet(opt['train_dataset_path'], resize_size=256)
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, num_workers=opt['num_workers'],
                              batch_size=opt['batch_size'], pin_memory=False)

    val_dataset = HEVCDataSet(opt['val_dataset_path'], opt['train_lambda'], 10, 'D', testfull=True, isTrain=False)
    val_loader = DataLoader(dataset=val_dataset, shuffle=False, num_workers=8, batch_size=2)

    """ logging """
    log = open(f'../main/saved_models/{experiment_name}/log_train.txt', 'a')
    log.write('Number of images: ' + str(train_dataset.__len__()) + '\n')
    log.close()
    print("Number of images: ", train_dataset.__len__())

    model = importlib.import_module('main.model.' + opt['model']).VideoCompressor().to(device)
    net = torch.nn.DataParallel(model).to(device)

    if opt['advanced_coder']:
        optimizer, aux_optimizer = configure_optimizers(opt, net)
    else:
        optimizer = optim.Adam(net.parameters(), lr=opt['lr'])

    log = open(f'../main/saved_models/{experiment_name}/log_train.txt', 'a')
    log.write(str(optimizer) + '\n')
    log.close()
    print("Optimizer: ", optimizer)

    if opt['load_model']:
        print("loading pretrain : ", opt['load_model'])
        model.load_state_dict(torch.load(opt['load_model'], map_location='cpu'), strict=True)

    """ start training """
    start_time = time.time()
    best_psnr = -1
    best_valid_psnr = -1
    best_bpp = 999
    best_valid_bpp = 999
    best_valid_msssim = -1
    iteration = 0  # each batch
    epoch = 0
    scaler = torch.cuda.amp.GradScaler()
    # from main.utils.utils import MS_SSIM
    # msssim_func = MS_SSIM(max_val=1.).cuda()

    while True:
        # adjust_learning_rate(optimizer, epoch, opt['lr'])
        net.train()
        sumloss, sumpsnr, sumbpp = [], [], []
        for batch_idx, input in enumerate(train_loader):

            lr = np.array([x['lr'] for x in optimizer.param_groups])[0]

            input_image, ref_image = Var(input[0]), Var(input[1])
            with torch.cuda.amp.autocast(enabled=enable_amp):
                recon_image, bpp_res, bpp_mv, mv_aux_loss, res_aux_loss = net(input_image, ref_image, enable_amp)
                mse_loss = torch.nn.MSELoss()(recon_image, input_image)
                # msssim = 1 - msssim_func(recon_image, input_image)

            bpp_res, bpp_mv, mv_aux_loss, res_aux_loss = torch.mean(bpp_res), torch.mean(bpp_mv), \
                                                         torch.mean(mv_aux_loss), torch.mean(res_aux_loss)
            aux_loss = mv_aux_loss + res_aux_loss
            bpp = bpp_res + bpp_mv
            # rd_loss = opt['train_lambda'] * msssim + bpp
            rd_loss = opt['train_lambda'] * mse_loss + bpp

            if enable_amp:
                optimizer.zero_grad()
                aux_optimizer.zero_grad()
                scaler.scale(rd_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
                scaler.step(optimizer)
                scaler.update()
                aux_loss.backward()
                aux_optimizer.step()
            else:
                optimizer.zero_grad()
                aux_optimizer.zero_grad()
                rd_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
                optimizer.step()
                aux_loss.backward()
                aux_optimizer.step()

            if iteration % 10 == 0:
                psnr = 10 * (torch.log(1 * 1 / mse_loss) / np.log(10)).cpu().detach().numpy() if mse_loss > 0 else 100
                sumloss.append(rd_loss.cpu().detach().numpy())
                sumpsnr.append(psnr)
                sumbpp.append(bpp.cpu().detach())

            """ logging and save models """
            if (iteration + 1) % 2000 == 0 or iteration == 0:
                torch.cuda.empty_cache()
                with open(f'../main/saved_models/{experiment_name}/log_train.txt', 'a') as log:
                    avg_rd_loss, avgpsnr, avgbpp = np.mean(sumloss), np.mean(sumpsnr), np.mean(sumbpp)

                    """ tensorboard record """
                    tb_logger.add_scalar('/train/lr', lr, global_step=iteration)
                    tb_logger.add_scalar('/train/rd_loss', rd_loss, global_step=iteration)
                    tb_logger.add_scalar('/train/avg_rd_loss', avg_rd_loss, global_step=iteration)
                    tb_logger.add_scalar('/train/avgpsnr', avgpsnr, global_step=iteration)
                    tb_logger.add_scalar('/train/avgbpp', avgbpp, global_step=iteration)

                    elapsed_time = time.time() - start_time
                    loss_log = f'[Epoch: {epoch} ({100. * batch_idx / len(train_loader):0.1f}%)] [{iteration + 1}/{opt["num_iter"]}] {"loss"}: {avg_rd_loss:0.3f}, {"elapsed_time"}: {elapsed_time:0.0f}, {"lr"}: {lr:0.5f}'

                    if avgpsnr > best_psnr:
                        best_psnr = avgpsnr
                    if avgbpp < best_bpp:
                        best_bpp = avgbpp

                    eval_log = f'{"bpp_mv":17s}: {bpp_mv:0.4f}, {"current_psnr":17s}: {avgpsnr:0.4f}, {"current_bpp":17s}: {avgbpp:0.4f}'
                    best_model_log = f'{"bpp_res":17s}: {bpp_res:0.4f}, {"best_psnr":17s}: {best_psnr:0.4f}, {"best_bpp":17s}: {best_bpp:0.4f}'

                    loss_model_log = f'{loss_log}\n{eval_log}\n{best_model_log}'
                    log.write(loss_model_log + '\n')
                    print(loss_model_log)

                sumloss, sumpsnr, sumbpp = [], [], []
                torch.save(model.state_dict(), f'../main/saved_models/{experiment_name}/latest.pth')

            # if True:
            if (iteration + 1) % 10000 == 0:
                """ iteration saving """
                torch.save(model.state_dict(),
                           "../main/saved_models/{}/{}_lambda{}.pth".format(experiment_name, iteration + 1,
                                                                            opt['train_lambda']))
                """ 
                start validation
                """
                torch.cuda.empty_cache()
                net.eval()
                with torch.no_grad():
                    eval_time = time.time()
                    valid_bpp, valid_mv, valid_res, valid_psnr, valid_msssim, valid_loss = validation(val_loader, net)

                    loss_log = f'[ Iteration Validation ] {"rd_loss"}: {valid_loss:0.6f}, {"cost_time"}: {time.time() - eval_time:0.0f}'
                    vaild_current_model_log = f'{"current_psnr":15s}: {valid_psnr:0.4f}, {"current_msssim":15s}: {valid_msssim:0.3f}, {"mv":5s}: {valid_mv:0.4f}, {"current_bpp":15s}: {valid_bpp:0.4f}'
                    if valid_psnr > best_valid_psnr:
                        best_valid_psnr = valid_psnr
                    if valid_bpp < best_valid_bpp:
                        best_valid_bpp = valid_bpp
                    if valid_msssim > best_valid_msssim:
                        best_valid_msssim = valid_msssim

                    vaild_best_model_log = f'{"best_psnr":15s}: {best_valid_psnr:0.4f}, {"best_msssim":15s}: {best_valid_msssim:0.3f}, {"res":5s}: {valid_res:0.4f}, {"best_bpp":15s}: {best_valid_bpp:0.4f}'
                    log = open(f'../main/saved_models/{experiment_name}/log_train.txt', 'a')
                    dashed_line = '-' * 90
                    vaild_model_log = f'{dashed_line}\n{loss_log}\n{vaild_current_model_log}\n{vaild_best_model_log}\n{dashed_line}'
                    print(vaild_model_log)
                    log.write(vaild_model_log + '\n')
                torch.cuda.empty_cache()
                net.train()

            iteration += 1
            # break
        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='../cfg/train.yaml', help='cfg')
    args = parser.parse_args()

    """ read cfg from yaml """
    opt = yaml.load(open(args.cfg, encoding='utf-8'), Loader=yaml.FullLoader)
    model_name = opt['model']
    enable_amp = opt['amp']

    experiment_name = opt['experiment_name'] + '_' + str(opt['train_lambda'])
    list(map(shutil.rmtree, glob.glob('../main/saved_models/{}'.format(experiment_name))))
    os.makedirs(f'../main/saved_models/{experiment_name}', exist_ok=True)
    os.system('cp ../main/model/' + opt['model'] + '.py' + ' ../main/saved_models/' + experiment_name)
    tb_logger = SummaryWriter(f'../main/saved_models/{experiment_name}/events')

    """ Seed and GPU setting """
    random.seed(1111)
    np.random.seed(1111)
    torch.manual_seed(1111)
    torch.cuda.manual_seed_all(1111)

    cudnn.benchmark = True
    cudnn.deterministic = True

    """ logging """
    log = open(f'../main/saved_models/{experiment_name}/log_train.txt', 'a')
    print(json.dumps(opt, indent=4, ensure_ascii=True))
    log.write(json.dumps(opt, indent=4, ensure_ascii=True) + '\n')
    log.close()

    train()
