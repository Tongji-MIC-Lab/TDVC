import os
import numpy as np
from get_data import get_data

QP = 22

yuv_path = '/dataset/video_cls/UCF101/UCF-101_fast_yuv_compress_' + str(QP)
out_path = '/dataset/video_cls/UCF101/UCF-101_fast_yuv_compress_' + str(QP) +'_img'

w, h = 320, 240

video_list = os.listdir(yuv_path)
video_ = get_data('/dataset/video_cls/UCF101/UCF101-json/ucf101_01.json')
# video_ = ['v_Biking_g23_c01']
for item in video_list:
    sub_list = os.listdir(os.path.join(yuv_path, item))
    video_object_path = os.path.join(out_path, item)
    for seq in sub_list:
        seq_name = os.path.basename(seq).split('.')[0].split('__')[0]
        if seq_name in video_:
            os.makedirs(os.path.join(video_object_path, seq_name), exist_ok=True)
            if '.log' in seq:
                with open(os.path.join(os.path.join(yuv_path, item), seq)) as f:
                    lines = f.readlines()

                size_line = []
                bppsum = 0
                for l in lines:
                    if ", size " in l:
                        size = l.split(',')[1]
                        size_line.append(int(size[5:]))
                        bppsum += int(size[5:])

                size_line = np.array(size_line) * 8.0 / (w * h)
                with open(os.path.join(video_object_path, seq_name, 'bpp.txt'), 'w', encoding='utf-8') as f:
                    for items in size_line:
                        f.write(str(items) + '\n')
            else:

                save_path = os.path.join(video_object_path, seq_name) + '/image_%05d.png'
                print('ffmpeg -y  ' + ' -i ' +
                      os.path.join(
                          os.path.join(yuv_path, item), seq) + ' ' + save_path)

                os.system('ffmpeg -y  ' + ' -i ' +
                      os.path.join(
                          os.path.join(yuv_path, item), seq) + ' ' + save_path)

                print()
