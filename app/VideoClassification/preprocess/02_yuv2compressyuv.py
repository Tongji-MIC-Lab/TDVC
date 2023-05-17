import os

from get_data import get_data

QP = 22
w, h = 320, 240
yuv_path = 'UCF101_yuv'
out_path = 'UCF-101_yuv_compress_' + str(QP)
os.makedirs(out_path, exist_ok=True)

keyint = 12

video_ = get_data('UCF101-json/ucf101_01.json')
# video_ = ['v_Biking_g23_c01']
for item in os.listdir(yuv_path):
    video_object_path = os.path.join(out_path, item)
    os.makedirs(video_object_path, exist_ok=True)
    for seq in os.listdir(os.path.join(yuv_path, item)):
        seq_name = os.path.basename(seq).split('.')[0]
        if seq_name in video_:
            print(
                'FFREPORT=file=' + video_object_path + '/' + seq_name + '.log:level=56 ffmpeg -y -pix_fmt yuv420p -s ' + str(
                    w) + 'x' + str(h) + ' -i ' + os.path.join(
                    os.path.join(yuv_path, item), seq) + ' -c:v libx265  -preset veryfast -tune zerolatency -x265-params "crf=' + str(
                    QP) + ':keyint=' + str(
                    keyint) + ':verbose=1" ' + ''.join([video_object_path, '/', seq_name, '__', str(QP), '__df.mkv']))

            os.system(
                'FFREPORT=file=' + video_object_path + '/' + seq_name + '.log:level=56 ffmpeg -y -pix_fmt yuv420p -s ' + str(
                    w) + 'x' + str(h) + ' -i ' + os.path.join(
                    os.path.join(yuv_path, item), seq) + ' -c:v libx265  -preset veryfast -tune zerolatency -x265-params "crf=' + str(
                    QP) + ':keyint=' + str(
                    keyint) + ':verbose=1" ' + ''.join([video_object_path, '/', seq_name, '__', str(QP), '__df.mkv']))
            # exit()
            print('-' * 30)
