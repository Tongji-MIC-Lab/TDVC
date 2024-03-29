import os

yuv_path = 'UCF101_yuv'
out_path = 'UCF101_yuv_img'

w, h = 320, 240

video_list = os.listdir(yuv_path)

for item in video_list:
    sub_list = os.listdir(os.path.join(yuv_path, item))
    video_object_path = os.path.join(out_path, item)

    for seq in sub_list:
        seq_name = os.path.basename(seq).split('.')[0]
        os.makedirs(os.path.join(video_object_path, seq_name), exist_ok=True)
        save_path = os.path.join(video_object_path, seq_name) + '/image_%05d.png'
        print('ffmpeg -y -pix_fmt yuv420p -s ' + str(w) + 'x' + str(h) + ' -i ' +
              os.path.join(
                  os.path.join(yuv_path, item), seq) + ' ' + save_path)
        #
        os.system('ffmpeg -y -pix_fmt yuv420p -s ' + str(w) + 'x' + str(h) + ' -i ' +
                  os.path.join(
                      os.path.join(yuv_path, item), seq) + ' ' + save_path)
