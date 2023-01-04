import glob
import os
import natsort

yuv_path = '/dataset/Subjective_Compare/ori/'
out_path = '/dataset/Subjective_Compare/ori_img2'

yuv_list = glob.glob(os.path.join(yuv_path, '*.mkv'))
yuv_list = natsort.natsorted(yuv_list)
for seq in yuv_list:
    seq_name = os.path.basename(seq).split('.')[0]
    print(seq_name)
    w, h = seq_name.split('_')[1].split('x')

    save_path = os.path.join(out_path, seq_name) + '/im%03d.png'
    os.makedirs(os.path.join(out_path, seq_name), exist_ok=True)

    # print('ffmpeg -y -pix_fmt yuv420p -s ' + str(w) + 'x' + str(h) + ' -i ' + seq + ' ' + save_path)
    # os.system('ffmpeg -y -pix_fmt yuv420p -s ' + str(w) + 'x' + str(h) + ' -i ' + seq + ' ' + save_path)

    print('ffmpeg -y' + ' -i ' + seq + ' ' + save_path)
    os.system('ffmpeg -y  ' + '-i ' + seq + ' ' + save_path)