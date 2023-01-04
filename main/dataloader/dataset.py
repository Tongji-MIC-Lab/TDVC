import glob
import os

import cv2
import natsort
import numpy as np
import torch
import torch.utils.data as data
from natsort import natsorted

from main.dataloader import augmentation
from main.model.basics import CalcuPSNR
from main.model.ms_ssim_torch import ms_ssim


class UVGDataSet(data.Dataset):
    def __init__(self, root, train_lambda, GOP_size, testfull=False, isTrain=True):
        self.inputPath = os.path.join(root, 'ori_img')
        self.refPath = os.path.join(root, 'compress_img_bpg')
        self.isTrain = isTrain
        self.ref = []
        self.refbpp = []
        self.input = []

        if train_lambda == 512 or train_lambda == 16:
            qp = 37
            # qp = 29
        elif train_lambda == 1024 or train_lambda == 32:
            qp = 32
            # qp = 26
        elif train_lambda == 2048 or train_lambda == 64:
            qp = 27
            # qp = 23
        elif train_lambda == 4096 or train_lambda == 128:
            qp = 22
            # qp = 20

        for idx, folder in enumerate(natsorted(os.listdir(self.inputPath))):
            seq = folder.rstrip()
            imglist = glob.glob(os.path.join(self.inputPath, seq, '*.png'))
            imglist = natsorted(imglist)
            if testfull:
                framerange = len(imglist) // GOP_size
            else:
                framerange = 8
            for i in range(framerange):
                refpath = os.path.join(self.refPath, seq, str(qp),
                                       'im' + str(i * GOP_size + 1).zfill(3) + '_' + str(qp) + '.png')
                with open(os.path.join(self.refPath, seq, str(qp),
                                       'im' + str(i * GOP_size + 1).zfill(3) + '_' + str(qp) + '.txt'), 'r',
                          encoding='utf-8') as f:
                    rbpp = f.read().splitlines()[0]
                inputpath = []
                for j in range(GOP_size):
                    inputpath.append(
                        os.path.join(self.inputPath, seq, 'im' + str(i * GOP_size + j + 1).zfill(3) + '.png'))
                self.ref.append(refpath)
                self.refbpp.append(float(rbpp))
                self.input.append(inputpath)



    def __len__(self):
        return len(self.ref)

    def __getitem__(self, index):
        ref_image_name = self.ref[index]

        ref_image = cv2.cvtColor(cv2.imread(ref_image_name), cv2.COLOR_BGR2RGB)
        if self.isTrain:
            ref_image = cv2.resize(ref_image, (256, 256))
        ref_image = ref_image.transpose(2, 0, 1).astype(np.float32) / 255.0

        h = ref_image.shape[1]
        w = ref_image.shape[2]

        ref_image = np.array(ref_image[:, :h, :w])
        input_images = []
        raw_video = []
        refpsnr = None
        refmsssim = None
        for filename in self.input[index]:
            input_image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
            if self.isTrain:
                input_image = cv2.resize(input_image, (256, 256))
            input_image = input_image.transpose(2, 0, 1)[:, :h, :w].astype(np.float32) / 255.0

            if refpsnr is None:
                refpsnr = CalcuPSNR(input_image, ref_image)
                refmsssim = ms_ssim(torch.from_numpy(input_image[np.newaxis, :]),
                                    torch.from_numpy(ref_image[np.newaxis, :]), data_range=1.0).numpy()
            else:
                # print('filename', filename)
                input_images.append(input_image[:, :h, :w])
            raw_video.append(input_image[:, :h, :w])
        raw_video = np.array(raw_video)
        input_images = np.array(input_images)
        return input_images, ref_image, self.refbpp[index], refpsnr, refmsssim, self.input[index], raw_video


class HEVCDataSet(data.Dataset):
    def __init__(self, root, train_lambda, GOP_size, cls, testfull=False, isTrain=True):
        self.inputPath = os.path.join(root, 'ori_img')
        self.refPath = os.path.join(root, 'compress_img_bpg')
        self.isTrain = isTrain
        self.ref = []
        self.refbpp = []
        self.input = []

        if cls == 'A':
            resolution = '2560x1600'
            seq_ = ['Traffic', 'PeopleOnStreet']
        elif cls == 'B':
            resolution = '1920x1080'
            seq_ = ['ParkScene', 'Kimono1', 'Cactus', 'BasketballDrive', 'BQTerrace']
        elif cls == 'C':
            resolution = '832x480'
            seq_ = ['BasketballDrill', 'BQMall', 'PartyScene', 'RaceHorses']
        elif cls == 'D':
            resolution = '416x240'
            seq_ = ['BasketballPass', 'BQSquare', 'BlowingBubbles', 'RaceHorses']
        elif cls == 'E':
            resolution = '1280x720'
            seq_ = ['vidyo1', 'vidyo3', 'vidyo4']

        if train_lambda == 512 or train_lambda == 16:
            qp = 37
            # qp = 29
        elif train_lambda == 1024 or train_lambda == 32:
            qp = 32
            # qp = 26
        elif train_lambda == 2048 or train_lambda == 64:
            qp = 27
            # qp = 23
        elif train_lambda == 4096 or train_lambda == 128:
            qp = 22
            # qp = 20

        for idx, folder in enumerate(os.listdir(self.inputPath)):
            seq = folder.rstrip()
            seq_r = seq.split('_')[1]
            seq_name = seq.split('_')[0]

            if seq_name in seq_ and seq_r == resolution:
                # print(seq)
                imglist = glob.glob(os.path.join(self.inputPath, seq, '*.png'))
                imglist = natsorted(imglist)
                if testfull:
                    framerange = len(imglist) // GOP_size
                else:
                    framerange = 8
                for i in range(framerange):
                    refpath = os.path.join(self.refPath, seq, str(qp),
                                           'im' + str(i * GOP_size + 1).zfill(3) + '_' + str(qp) + '.png')
                    with open(os.path.join(self.refPath, seq, str(qp),
                                           'im' + str(i * GOP_size + 1).zfill(3) + '_' + str(qp) + '.txt'), 'r',
                              encoding='utf-8') as f:
                        rbpp = f.read().splitlines()[0]
                    inputpath = []
                    for j in range(GOP_size):
                        inputpath.append(
                            os.path.join(self.inputPath, seq, 'im' + str(i * GOP_size + j + 1).zfill(3) + '.png'))
                    self.ref.append(refpath)
                    self.refbpp.append(float(rbpp))
                    self.input.append(inputpath)

    def __len__(self):
        return len(self.ref)

    def __getitem__(self, index):
        ref_image_name = self.ref[index]

        ref_image = cv2.cvtColor(cv2.imread(ref_image_name), cv2.COLOR_BGR2RGB)
        if self.isTrain:
            ref_image = cv2.resize(ref_image, (256, 256))
        ref_image = ref_image.transpose(2, 0, 1).astype(np.float32) / 255.0

        h = (ref_image.shape[1])
        w = (ref_image.shape[2])

        ref_image = np.array(ref_image[:, :h, :w])
        raw_video = []
        input_images = []
        refpsnr = None
        refmsssim = None
        for filename in self.input[index]:
            input_image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
            if self.isTrain:
                input_image = cv2.resize(input_image, (256, 256))
            input_image = input_image.transpose(2, 0, 1)[:, :h, :w].astype(np.float32) / 255.0

            if refpsnr is None:
                refpsnr = CalcuPSNR(input_image, ref_image)
                refmsssim = ms_ssim(torch.from_numpy(input_image[np.newaxis, :]),
                                    torch.from_numpy(ref_image[np.newaxis, :]), data_range=1.0).numpy()
            else:
                input_images.append(input_image[:, :h, :w])
            raw_video.append(input_image[:, :h, :w])
        input_images = np.array(input_images)
        raw_video = np.array(raw_video)
        return input_images, ref_image, self.refbpp[index], refpsnr, refmsssim, ref_image_name, raw_video


class DataSet(data.Dataset):
    def __init__(self, dataset_path, resize_size):
        self.image_input_list, self.image_ref_list = self.get_vimeo(dataset_path)
        self.size = [resize_size, resize_size]
        self.im_height, self.im_width = self.size[0], self.size[1]


    def get_vimeo(self, dataset_path):
        fns_train_input = []
        fns_train_ref = []
        dirs = natsort.natsorted(os.listdir(dataset_path))
        # dirs = ['00017', '00043', '00025', '00094', '00033', '00067', '00073', '00047', '00021', '00029', '00058', '00019', '00088', '00045', '00063', '00036', '00055', '00049', '00064', '00057']
        for dir in dirs:
            sub_dirs = natsort.natsorted(os.listdir(os.path.join(dataset_path, dir)))
            for file in sub_dirs:
                file_list = glob.glob(os.path.join(dataset_path, dir, file, '*.png'))
                start = 1
                end = len(file_list)
                while True:
                    interval = 1
                    if start + interval <= end:
                        tmp_list = [os.path.join(dataset_path, dir, file, f'im{1}.png')]

                        for i in range(max(start + interval - 3, 1), start + interval):
                            tmp_list.append(os.path.join(dataset_path, dir, file, f'im{i}.png'))

                        if len(tmp_list) != 4:
                            for _ in range(len(tmp_list), 4):
                                tmp_list.append(tmp_list[-1])

                        fns_train_ref.append(tmp_list)
                        fns_train_input.append(os.path.join(dataset_path, dir, file, f'im{start + interval}.png'))
                        start = start + 1
                    else:
                        break

                tmp_list = [os.path.join(dataset_path, dir, file, f'im{1}.png')]
                tmp_list.append(os.path.join(dataset_path, dir, file, f'im{1}.png'))
                tmp_list.append(os.path.join(dataset_path, dir, file, f'im{3}.png'))
                tmp_list.append(os.path.join(dataset_path, dir, file, f'im{5}.png'))
                fns_train_ref.append(tmp_list)
                fns_train_input.append(os.path.join(dataset_path, dir, file, f'im{7}.png'))

        return fns_train_input, fns_train_ref

    def __len__(self):
        return len(self.image_input_list)

    def __getitem__(self, index):
        input_image = cv2.cvtColor(cv2.imread(self.image_input_list[index]), cv2.COLOR_BGR2RGB)
        ref_image = []
        for idx, i in enumerate(self.image_ref_list[index]):
            ref_image.append(cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2RGB))
        input_image, ref_images = augmentation.imgauglist2(input_image, ref_image, self.size)

        return input_image, ref_images