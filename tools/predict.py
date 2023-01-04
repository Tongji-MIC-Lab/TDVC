import argparse
import importlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
from rich.progress import track
from torch.autograd import Variable
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
from main.dataloader.dataset import UVGDataSet, HEVCDataSet
from main.model.ms_ssim_torch import ms_ssim
from main.utils.utils import MS_SSIM
from main.utils.utils import crop, pad

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cudnn.benchmark = True
cudnn.deterministic = True


def Var(x):
    return Variable(x.to(device))


msssim_func = MS_SSIM(max_val=1.).cuda()


def validation(val_loader, net):
    st = time.time()
    log = open(f'../main/test_dir/{opt["val_dataset"]}_{root_path}/{opt["class"]}_{save_name}.txt', 'w')
    data_len_log = f'number of test images: \n{len(test_dataset.input)} × {GOP_size}\n'
    print(data_len_log)
    print(json.dumps(opt, indent=4, ensure_ascii=True))

    psnrs, bpps, ssims, mvs, ress, mses = [], [], [], [], [], []
    for batch_idx, input in track(enumerate(val_loader), total=len(val_loader)):
        input_images, ref_image, ref_bpp, ref_psnr, ref_msssim = input[0], input[1], input[2], input[3], input[4]
        input_image_name = input[5]
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
            # seti = time.time()
            with torch.cuda.amp.autocast(enabled=enable_amp):
                recon_image, bpp_res, bpp_mv = net(inputframe, refframes_, enable_amp)
            # print((time.time() - seti) * 1000, 'ms')

            ref_imglist.append(recon_image)
            recon_image = crop(recon_image, (ref_image.size(2), ref_image.size(3)))
            inputframe = crop(inputframe, (ref_image.size(2), ref_image.size(3)))

            # x = np.uint8(recon_image.squeeze(0).mul(255).detach().cpu().numpy().transpose(1, 2, 0))
            # plt.figure(figsize=(10, 10))
            # plt.axis('off')
            # plt.imshow(x)
            # plt.show()

            # import cv2
            # save_namep = 'e2e_compress2048_img_psnr_ori'
            # os.makedirs(os.path.dirname(input_image_name[i+1][0]).replace('ori_img', save_namep), exist_ok=True)
            # x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
            # cv2.imwrite(input_image_name[i+1][0].replace('ori_img', save_namep), x)
            # with open(os.path.dirname(input_image_name[i+1][0]).replace('ori_img', save_namep) +'bpp.txt', 'a', encoding='utf-8') as f:
            #     f.write(input_image_name[i+1][0].replace('ori_img', save_namep) + '\t' +str((bpp_res + bpp_mv)[0].cpu().detach().numpy()) + '\n')

            for b in range(recon_image.size(0)):
                mse_loss = torch.nn.MSELoss()(recon_image[b], inputframe[b])
                psnr = float(10 * (torch.log(1. / mse_loss) / np.log(10)).cpu().detach().numpy())
                psnrs.append(psnr)
                bpps.append(float((bpp_res + bpp_mv)[b].cpu().detach().numpy()))

                if enable_amp:
                    ssims.append(float(ms_ssim(recon_image[b].float().unsqueeze(0), inputframe[b].unsqueeze(0),
                                               data_range=1.0).cpu().detach().numpy()))
                else:
                    ssims.append(msssim_func(recon_image[b].unsqueeze(0), inputframe[b].unsqueeze(0)))

                mvs.append(float(bpp_mv[b]))
                ress.append(float(bpp_res[b]))
                mses.append(float(mse_loss))

    result_info = "bpp : %.6lf\n\npsnr : %.6lf\n\nmsssim : %.6lf" % (np.mean(bpps), np.mean(psnrs), np.mean(ssims))
    log.write(result_info)
    log.write(data_len_log)
    log.write('cfg :\n' + json.dumps(opt, indent=4, ensure_ascii=True) + '\n')
    log.write('cost_time :\n' + str(time.time() - st) + '\n')
    print(result_info)
    log.close()

    return np.mean(bpps), np.mean(mvs), np.mean(ress), np.mean(psnrs), np.mean(ssims), np.mean(mses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', help='pretrain')
    parser.add_argument('--val_dataset', help='pretrain')
    parser.add_argument('--l', help='lambda')
    parser.add_argument('--cls', help='HEVC_cls')
    parser.add_argument('--cfg', default='../cfg/predict.yaml', help='HEVC_cls')

    args = parser.parse_args()

    """ read cfg from yaml """
    opt = yaml.load(open(args.cfg, encoding='utf-8'), Loader=yaml.FullLoader)

    if args.pretrain:
        opt['pretrain'] = args.pretrain

    # 上一级目录加上文件名
    root_path = os.path.basename(os.path.abspath(os.path.dirname(opt['pretrain'])))
    lambda_ = os.path.basename(opt['pretrain']).split('_')[1].split('.')[0].replace('lambda', '')

    if args.l:
        lambda_ = args.l
    print(lambda_)

    if args.cls:
        opt["class"] = args.cls

    if args.val_dataset:
        opt["val_dataset"] = args.val_dataset

    save_name = os.path.basename(opt['pretrain']).replace('.pth', '')
    os.makedirs(f'../main/test_dir/{opt["val_dataset"]}_{root_path}', exist_ok=True)

    enable_amp = opt['enable_amp']
    model_ = importlib.import_module('main.model.' + opt['model'])
    net = model_.VideoCompressor().to(device)

    net.load_state_dict(torch.load(opt['pretrain'], map_location='cpu'), strict=True)

    net = torch.nn.DataParallel(net).to(device)

    if opt['val_dataset'] == 'UVG':
        GOP_size = 12
        test_dataset = UVGDataSet('/dataset/UVG2/', int(lambda_), GOP_size, testfull=True,
                                  isTrain=False)
    elif opt['val_dataset'] == 'MCL-JCV':
        GOP_size = 12
        test_dataset = UVGDataSet('/dataset/MCL-JCV/', int(lambda_), GOP_size, testfull=True,
                                  isTrain=False)
    else:
        GOP_size = 10
        test_dataset = HEVCDataSet('/dataset/HEVC2/', int(lambda_), GOP_size, opt['class'],
                                   testfull=True, isTrain=False)

    test_loader = DataLoader(dataset=test_dataset, num_workers=4, batch_size=1)

    st = time.time()
    net.eval()
    with torch.no_grad():
        valid_bpp, valid_mv, valid_res, valid_psnr, valid_msssim, valid_loss = validation(test_loader, net)
    print("\ncost_time : %.2lf" % (time.time() - st))
