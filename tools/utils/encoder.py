import struct
import time
from pathlib import Path
import os
import sys

from tools.utils.getpsnr import psnr

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
from main.model.net import VideoCompressor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cudnn.benchmark = True
cudnn.deterministic = True


def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size


ref_image = cv2.cvtColor(
    cv2.imread('/mnt/dataset/UVG/compress_img_bpg/Beauty_1920x1080_120fps_420_8bit_YUV/22/im001_22.png'),
    cv2.COLOR_BGR2RGB)
ref_image = cv2.resize(ref_image, (256, 256))
ref_image = ref_image.transpose(2, 0, 1).astype(np.float32) / 255.0

input_image = cv2.cvtColor(cv2.imread('/mnt/dataset/UVG/ori_img/Beauty_1920x1080_120fps_420_8bit_YUV/im002.png'),
                           cv2.COLOR_BGR2RGB)
input_image = cv2.resize(input_image, (256, 256))
input_image_ = input_image.copy()
input_image = input_image.transpose(2, 0, 1).astype(np.float32) / 255.0
shape = input_image.shape

refframe = torch.from_numpy(ref_image).unsqueeze(0).to(device)
inputframe = torch.from_numpy(input_image).unsqueeze(0).to(device)

net = VideoCompressor(True).to(device)
net.load_state_dict(
    torch.load('xxx', map_location='cpu'),
    strict=False)
net.eval()

enc_start = time.time()
output_dict = net(inputframe, refframe, False)
clipped_recon_image, mse_loss, out_string, out_string_shape, bpp_res, bpp_mv = output_dict['clipped_recon_image'], \
                                                                               output_dict['mse_loss'], output_dict[
                                                                                   'out_string'], output_dict[
                                                                                   'out_string_shape'], output_dict[
                                                                                   'bpp_res'], output_dict['bpp_mv']
real_bpp_mv, real_bpp_res = output_dict['real_bpp_mv'], output_dict['real_bpp_res']

with open('./save.bin', 'wb') as f:
    for idx, s in enumerate(out_string):
        # 额外开销
        values = out_string_shape[idx]
        f.write(struct.pack(">{:d}I".format(len(values)), *values))
        f.write(np.array(len(s), dtype=np.uint16).tobytes())
        # 真实bpp
        f.write(s)

enc_time = time.time() - enc_start

bpp = float(filesize('./save.bin')) * 8 / (shape[1] * shape[2])
print(
    f"torch psnr {torch.mean(10 * (torch.log(1. / mse_loss) / np.log(10))).cpu().detach().numpy()}  \n"
    f"cv2 psnr {psnr(input_image_, clipped_recon_image)}  \n"
    f"calc bpp {(bpp_res + bpp_mv).cpu().detach().numpy()}  \n"
    f"real bpp {(real_bpp_mv + real_bpp_res).cpu().detach().numpy()[0]} \n"
    f"file bpp {bpp}  \n"
    f"Encoded in {enc_time:.2f}s"
)
