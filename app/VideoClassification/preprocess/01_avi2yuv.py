import os

filepath = 'UCF101'
objectpath = filepath + '_yuv'

os.makedirs(objectpath, exist_ok=True)

video_list = os.listdir(filepath)

for item in video_list:
    sub_list = os.listdir(os.path.join(filepath, item))
    video_object_path = os.path.join(objectpath, item)
    os.makedirs(video_object_path, exist_ok=True)
    for subitem in sub_list:
        os.system('ffmpeg -i ' + os.path.join(os.path.join(filepath, item), subitem) + ' ' + os.path.join(video_object_path, subitem.replace('.avi', '.yuv')))

