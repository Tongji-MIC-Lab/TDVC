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
# qp = 37
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
        filename = 'im' + b + '.png'
        filename_gop.append(filename)

    for each_img in img_list:
        seq_name = each_img.split('/')[-2]
        w, h = seq_name.split('_')[1].split('x')

        filename = os.path.basename(each_img)
        if filename in filename_gop:
            bin_path = save_path + '/' + filename.replace('.png', '') + '_' + str(qp) + '.bin'

            bits = os.path.getsize(bin_path)
            bits = bits * 8

            with open(bin_path.replace('.bin', '.txt'), 'w', encoding='utf-8') as f:
                f.write(str(bits / int(w) / int(h)) + '\n')
