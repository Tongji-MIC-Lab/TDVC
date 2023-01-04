import glob
import os
import sys
from natsort import natsorted

ori_img_path = '/dataset/Subjective_Compare/ori_img'
compress_path = '/dataset/Subjective_Compare/compress_img_bpg'

os.makedirs(compress_path, exist_ok=True)

ori_img_list = os.listdir(ori_img_path)
ori_img_list = natsorted(ori_img_list)

qp = sys.argv[1]
# qp = 22
GOP = 12

for dir in ori_img_list:
    img_list = glob.glob(os.path.join(ori_img_path, dir, '*.png'))
    save_path = os.path.join(compress_path, dir, str(qp))
    os.makedirs(save_path, exist_ok=True)
    img_list = natsorted(img_list)

    filename_gop = []
    framerange = len(img_list) // GOP
    for i in range(framerange):
        b = str(i * GOP + 1).zfill(3)
        if int(b) < len(img_list):
            filename = 'im' + b + '.png'
            filename_gop.append(filename)

    for each_img in img_list:
        filename = os.path.basename(each_img)
        if filename in filename_gop:
            bin_path = save_path + '/' + filename.replace('.png', '') + '_' + str(qp) + '.bin'
            # print('bpgdec ' + bin_path + ' -o ' + bin_path.replace('.bin', '.png'))
            os.system('bpgdec ' + bin_path + ' -o ' + bin_path.replace('.bin', '.png'))
