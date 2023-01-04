import glob
import os
import natsort
import re

import numpy

import glob
import math
import os

import cv2
import numpy as np
from skimage.measure import compare_ssim

mkv_name = '265compress'
mkvimg_name = mkv_name + '_img'
yuv_path = '/dataset/Subjective_Compare/' + mkv_name + '/'
out_path = '/dataset/Subjective_Compare/' + mkvimg_name

os.makedirs(yuv_path, exist_ok=True)
os.makedirs(yuv_path.replace('265', '264'), exist_ok=True)
filedir = 'videoSRC18_1920x1080_25'
filedir = 'vidyo1_1280x720_60'

os.system('FFREPORT=file=/dataset/Subjective_Compare/' + mkv_name.replace('265', '264') + '/vidyo1_1280x720_60.log:level=56 ffmpeg -y -pix_fmt yuv420p  -framerate 60 -s 1280x720 -i /dataset/Subjective_Compare/ori/vidyo1_1280x720_60.yuv -c:v libx264 -tune zerolatency -qp 35 -g 12 -bf 2 -b_strategy 0 -sc_threshold 0 /dataset/Subjective_Compare/' + mkv_name.replace('265', '264') + '/vidyo1_1280x720_60.mkv')
# os.system('FFREPORT=file=/dataset/Subjective_Compare/' + mkv_name.replace('265', '264') + '/Bosphorus_1920x1080_120fps_420_8bit_YUV.log:level=56 ffmpeg -y -pix_fmt yuv420p  -framerate 120 -s 1920x1080 -i /dataset/Subjective_Compare/ori/Bosphorus_1920x1080_120fps_420_8bit_YUV.yuv -c:v libx264 -tune zerolatency -qp 33 -g 12 -bf 2 -b_strategy 0 -sc_threshold 0 /dataset/Subjective_Compare/' + mkv_name.replace('265', '264') + '/Bosphorus_1920x1080_120fps_420_8bit_YUV.mkv')
# os.system('FFREPORT=file=/dataset/Subjective_Compare/' + mkv_name.replace('265', '264') + '/videoSRC18_1920x1080_25.log:level=56 ffmpeg -y -pix_fmt yuv420p  -framerate 25 -s 1920x1080 -i /dataset/Subjective_Compare/ori/videoSRC18_1920x1080_25.yuv -c:v libx264 -tune zerolatency -qp 33 -g 12 -bf 2 -b_strategy 0 -sc_threshold 0 /dataset/Subjective_Compare/' + mkv_name.replace('265', '264') + '/videoSRC18_1920x1080_25.mkv')

os.system('FFREPORT=file=/dataset/Subjective_Compare/' + mkv_name + '/vidyo1_1280x720_60.log:level=56 ffmpeg -y -pix_fmt yuv420p  -framerate 60 -s 1280x720 -i /dataset/Subjective_Compare/ori/vidyo1_1280x720_60.yuv -c:v libx265  -preset veryfast -tune zerolatency -x265-params "qp=29" /dataset/Subjective_Compare/' + mkv_name + '/vidyo1_1280x720_60.mkv')
# os.system('FFREPORT=file=/dataset/Subjective_Compare/' + mkv_name + '/Bosphorus_1920x1080_120fps_420_8bit_YUV.log:level=56 ffmpeg -y -pix_fmt yuv420p  -framerate 120 -s 1920x1080 -i /dataset/Subjective_Compare/ori/Bosphorus_1920x1080_120fps_420_8bit_YUV.yuv -c:v libx265  -preset veryfast -tune zerolatency -x265-params "qp=28" /dataset/Subjective_Compare/' + mkv_name + '/Bosphorus_1920x1080_120fps_420_8bit_YUV.mkv')
# os.system('FFREPORT=file=/dataset/Subjective_Compare/' + mkv_name + '/videoSRC18_1920x1080_25.log:level=56 ffmpeg -y -pix_fmt yuv420p  -framerate 25 -s 1920x1080 -i /dataset/Subjective_Compare/ori/videoSRC18_1920x1080_25.yuv -c:v libx265  -preset veryfast -tune zerolatency -x265-params "qp=30" /dataset/Subjective_Compare/' + mkv_name + '/videoSRC18_1920x1080_25.mkv')


for seq in natsort.natsorted(glob.glob(os.path.join(yuv_path, filedir + '.*'))):
    seq_name = os.path.basename(seq).split('.')[0]
    print(seq_name)
    w, h = seq_name.split('_')[1].split('x')
    save_path = os.path.join(out_path, seq_name) + '/im%03d.png'
    os.makedirs(os.path.join(out_path, seq_name), exist_ok=True)
    # print('ffmpeg -y -pix_fmt yuv420p -s ' + str(w) + 'x' + str(h) + ' -i ' + seq + ' ' + save_path)
    # os.system('ffmpeg -y -pix_fmt yuv420p -s ' + str(w) + 'x' + str(h) + ' -i ' + seq + ' ' + save_path)
    print('ffmpeg -y ' + ' -i ' + seq + ' ' + save_path)
    os.system('ffmpeg -y  ' + '-i ' + seq + ' ' + save_path)

for seq in natsort.natsorted(glob.glob(os.path.join(yuv_path.replace('265', '264'), filedir + '.*'))):
    seq_name = os.path.basename(seq).split('.')[0]
    print(seq_name)
    w, h = seq_name.split('_')[1].split('x')
    save_path = os.path.join(out_path.replace('265', '264'), seq_name) + '/im%03d.png'
    os.makedirs(os.path.join(out_path.replace('265', '264'), seq_name), exist_ok=True)
    # print('ffmpeg -y -pix_fmt yuv420p -s ' + str(w) + 'x' + str(h) + ' -i ' + seq + ' ' + save_path)
    # os.system('ffmpeg -y -pix_fmt yuv420p -s ' + str(w) + 'x' + str(h) + ' -i ' + seq + ' ' + save_path)
    print('ffmpeg -y ' + ' -i ' + seq + ' ' + save_path)
    os.system('ffmpeg -y  ' + '-i ' + seq + ' ' + save_path)


root_path = '/dataset/Subjective_Compare'

def psnr(img1, img2):
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 10 * math.log10(PIXEL_MAX**2 / mse)



def main(file):
    # im_width = float(1920)
    # im_height = float(1080)

    im_width = float(1280)
    im_height = float(720)

    with open(file) as f:
        lines = f.readlines()

    size_line = []
    bppsum = 0
    for l in lines:
        if ", size " in l:
            size = l.split(',')[1]
            size_line.append(int(size[5:]))
            bppsum += int(size[5:])

    size_line = numpy.array(size_line) * 8.0 / (im_width * im_height)



    bppm = numpy.mean(size_line)
    # print(bppm)

    bpp_str = ''
    for l in lines:
        if "Lsize" in l:
            bpp_str = l
            break

    bpp_strs = re.findall(r"[-+]?\d*\.\d+|\d+", bpp_str)
    bpp = float(float(bpp_strs[6]) * float(bpp_strs[7])) * 1000 / (float(bpp_strs[0]) * im_width * im_height)

    # print(bpp,'\n')
    return size_line[100]



print('x264_bpp', main('/dataset/Subjective_Compare/' + mkv_name.replace('265', '264') + '/' + filedir + '.log'))
print('x265_bpp', main('/dataset/Subjective_Compare/' + mkv_name + '/' + filedir + '.log'))
raw_img = cv2.imread(os.path.join(root_path, 'ori_img', filedir, 'im' + '{:03d}'.format(100) + '.png'))
x264_img = cv2.imread(os.path.join(root_path, mkvimg_name.replace('265', '264'), filedir, 'im' + '{:03d}'.format(100) + '.png'))
x265_img = cv2.imread(os.path.join(root_path, mkvimg_name, filedir, 'im' + '{:03d}'.format(100) + '.png'))
proposed_img = cv2.imread(os.path.join(root_path, 'e2e_compress2048_img_psnr', filedir, 'im' + '{:03d}'.format(100) + '.png'))
print(psnr(raw_img, x264_img))
print(psnr(raw_img, x265_img))
print(psnr(raw_img, proposed_img))
