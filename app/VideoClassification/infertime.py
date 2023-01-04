import glob
import os
import sys

import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
import torch.nn as nn
import torchvision
import time
import torchvision.transforms as transforms
from dataloader import presets
from thop import profile
from thop import clever_format
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
from main.model.pnet_cls import VideoCompressor
from app.VideoClassification.dataloader.presets import VideoClassificationPresetTrain_cls, \
    VideoClassificationPresetEval_cls, pad, crop
from app.VideoClassification.models.decompress.basicvsr_pp import BasicVSRPlusPlus

class MODEL(nn.Module):
    def __init__(self, compress, ehc=False):
        super(MODEL, self).__init__()
        self.compress = compress
        self.ehc = ehc
        self.clip = 8
        self.videocls = torchvision.models.video.__dict__['r2plus1d_18'](pretrained=True)
        self.videocompress = VideoCompressor()
        self.videocompress.load_state_dict(torch.load('xxx', map_location='cpu'))

        self.trans = VideoClassificationPresetTrain_cls()
        self.trans_eval = VideoClassificationPresetEval_cls()

        self.ehc_model = BasicVSRPlusPlus(mid_channels=128, num_blocks=25).cuda()
        bd = torch.load('xxx',
            map_location='cpu')['state_dict']
        new_dict = {}
        for k, v in bd.items():
            new_dict.setdefault(k.replace('generator.', ''), v)
        self.ehc_model.load_state_dict(new_dict, False)

    def forward(self, ref_video, raw_video, amp):
        output2 = None
        psnr_c, psnr_h, sumbpp = [], [], []
        if self.compress:
            video_crop = [ref_video[:, :, 0, :, :]]
            rh, rw = ref_video[:, :, 0, :, :].size(2), ref_video[:, :, 0, :, :].size(3)
            ref_images = [pad(ref_video[:, :, 0, :, :], 64)]
            for i in range(raw_video.size()[2] - 1):
                input_image = pad(raw_video[:, :, i + 1, :, :], 64)
                if len(ref_images) == 1:
                    ref_frames = torch.stack([ref_images[0], ref_images[-1], ref_images[-1], ref_images[-1]])
                elif len(ref_images) == 2:
                    ref_frames = torch.stack([ref_images[0], ref_images[-2], ref_images[-1], ref_images[-1]])
                else:
                    ref_frames = torch.stack([ref_images[0], ref_images[-3], ref_images[-2], ref_images[-1]])
                recon_image, bpp = self.videocompress(input_image, ref_frames.view(-1, 4, 3, input_image.size(2),
                                                                                   input_image.size(3)), amp)
                sumbpp.append(bpp)
                ref_images.append(recon_image)
                video_crop.append(crop(recon_image, (rh, rw)))
            video = torch.stack(video_crop, dim=2)
            if self.ehc:
                print('ehc')
                output2 = self.ehc_model(video.transpose(2, 1).contiguous()).clamp(0., 1.).transpose(2, 1)

        else:
            video = ref_video
        B, C, N, H, W = video.size()
        video = video.permute(0, 2, 1, 3, 4).contiguous().view(-1, C, H, W)  # B N C H W
        videoc = self.trans_eval(video)
        video = videoc.view(B, N, C, videoc.size(2), videoc.size(3)).permute(0, 2, 1, 3, 4)

        with torch.cuda.amp.autocast(enabled=False):
            output = self.videocls(video)
        return output, output2


if __name__ == "__main__":
    # model = MODEL(True, True).cuda()
    # flops, params = profile(model, inputs=(torch.randn(1, 3, 8, 320, 240).cuda(), torch.randn(1, 3, 8, 320, 240).cuda(), True))
    # print(flops)
    # print(params)
    # flops, params = clever_format([flops, params], "%.3f")
    # print(flops)
    # print(params)
    #
    # exit()

    times = []
    model = MODEL(False).cuda()
    model.eval()
    model.videocls.fc = nn.Linear(512, 101).cuda()
    dicts = torch.load('xxx', map_location='cpu')['model']
    new_dict = {}
    for k, v in dicts.items():
        if 'videocls' in k:
            new_dict.setdefault(k, v)
    model.load_state_dict(new_dict, False)
    s = time.time()
    os.system('ffmpeg -y  -pix_fmt yuv420p -s 320x240 -i /dataset/video_cls/Speed_Test/v_Biking_g01_c01.yuv -c:v libx265  -vframes 8 -preset veryfast -tune zerolatency -x265-params "crf=22:keyint=12:verbose=1" /dataset/video_cls/Speed_Test/compress/v_Biking_g01_c01_22.mkv')
    os.system('ffmpeg -y -i /dataset/video_cls/Speed_Test/compress/v_Biking_g01_c01_22.mkv  /dataset/video_cls/Speed_Test/compressimg/image_%05d.png')
    x265 = [cv2.imread(x) for x in glob.glob('/dataset/video_cls/Speed_Test/compressimg/*.png')]
    clip = [transforms.ToTensor()(img) for img in x265]
    clip = torch.stack(clip, 0).permute(0, 2, 3, 1)
    clip = presets.VideoClassificationPresetEval()(clip).unsqueeze(0)
    with torch.no_grad():
        output, output2 = model(clip.cuda(), clip.cuda(), amp=False)
        # _, pred = output.topk(5, 1, True, True)
    t1 = time.time() - s
    times.append(('t1', t1))
    # print(pred)


    model = MODEL(True).cuda()
    model.eval()
    model.videocls.fc = nn.Linear(512, 101).cuda()
    s = time.time()
    os.system('ffmpeg -y  -pix_fmt yuv420p -s 320x240 -i /dataset/video_cls/Speed_Test/v_Biking_g01_c01.yuv -c:v libx265  -vframes 1 -preset veryfast -tune zerolatency -x265-params "crf=22:keyint=12:verbose=1" /dataset/video_cls/Speed_Test/compress/v_Biking_g01_c01_22.mkv')
    os.system('ffmpeg -y -i /dataset/video_cls/Speed_Test/compress/v_Biking_g01_c01_22.mkv -vframes 1 /dataset/video_cls/Speed_Test/compressimg/image_%05d.png')
    os.system('ffmpeg -y -pix_fmt yuv420p -s 320x240 -i  /dataset/video_cls/Speed_Test/v_Biking_g01_c01.yuv -vframes 7 /dataset/video_cls/Speed_Test/yuvimg//image_%05d.png')
    cp = [cv2.imread(x) for x in glob.glob('/dataset/video_cls/Speed_Test/compressimg/*.png')]
    clip = [transforms.ToTensor()(img) for img in cp]
    clip = torch.stack(clip, 0).permute(0, 2, 3, 1)
    clip = presets.VideoClassificationPresetEval()(clip).unsqueeze(0)
    with torch.no_grad():
        model(clip.cuda(), clip.cuda(), amp=False)
    t2 = time.time() - s
    times.append(('t2', t2))


    model = MODEL(True, True).cuda()
    model.eval()
    model.videocls.fc = nn.Linear(512, 101).cuda()
    s = time.time()
    os.system('ffmpeg -y  -pix_fmt yuv420p -s 320x240 -i /dataset/video_cls/Speed_Test/v_Biking_g01_c01.yuv -c:v libx265  -vframes 1 -preset veryfast -tune zerolatency -x265-params "crf=22:keyint=12:verbose=1" /dataset/video_cls/Speed_Test/compress/v_Biking_g01_c01_22.mkv')
    os.system('ffmpeg -y -i /dataset/video_cls/Speed_Test/compress/v_Biking_g01_c01_22.mkv -vframes 1 /dataset/video_cls/Speed_Test/compressimg/image_%05d.png')
    os.system('ffmpeg -y -pix_fmt yuv420p -s 320x240 -i  /dataset/video_cls/Speed_Test/v_Biking_g01_c01.yuv -vframes 7 /dataset/video_cls/Speed_Test/yuvimg//image_%05d.png')
    cp = [cv2.imread(x) for x in glob.glob('/dataset/video_cls/Speed_Test/compressimg/*.png')]
    clip = [transforms.ToTensor()(img) for img in cp]
    clip = torch.stack(clip, 0).permute(0, 2, 3, 1)
    clip = presets.VideoClassificationPresetEval()(clip).unsqueeze(0)
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            model(clip.cuda(), clip.cuda(), amp=True)
    t3 = time.time() - s
    times.append(('t3', t3))

    print(times)


