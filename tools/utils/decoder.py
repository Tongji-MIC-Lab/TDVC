import struct
import time
from pathlib import Path
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
from main.model.decNet.net_src import VideoCompressor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cudnn.benchmark = True
cudnn.deterministic = True


def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size


string_, string_shape = [], []
with open('./save.bin', 'rb') as f:
    shape = struct.unpack(">{:d}I".format(4), f.read(4 * struct.calcsize("I")))
    mv_len = np.frombuffer(f.read(2), dtype=np.uint16)
    string_si = f.read(np.int(mv_len))
    string_.append(string_si)
    string_shape.append(shape)

    shape = struct.unpack(">{:d}I".format(4), f.read(4 * struct.calcsize("I")))
    mv_len = np.frombuffer(f.read(2), dtype=np.uint16)
    string_z = f.read(np.int(mv_len))
    string_.append(string_z)
    string_shape.append(shape)

    shape = struct.unpack(">{:d}I".format(4), f.read(4 * struct.calcsize("I")))
    mv_len = np.frombuffer(f.read(2), dtype=np.uint16)
    string_mv = f.read(np.int(mv_len))
    string_.append(string_mv)
    string_shape.append(shape)

ref_image = cv2.cvtColor(
    cv2.imread('/dataset/UVG/compress_img_bpg/Beauty_1920x1080_120fps_420_8bit_YUV/22/im001_22.png'),
    cv2.COLOR_BGR2RGB)
ref_image = cv2.resize(ref_image, (256, 256))
ref_image = ref_image.transpose(2, 0, 1).astype(np.float32) / 255.0

input_image = cv2.cvtColor(cv2.imread('/dataset/UVG/ori_img/Beauty_1920x1080_120fps_420_8bit_YUV/im002.png'),
                           cv2.COLOR_BGR2RGB)
input_image = cv2.resize(input_image, (256, 256))
input_image = input_image.transpose(2, 0, 1).astype(np.float32) / 255.0
shape = input_image.shape

refframe = torch.from_numpy(ref_image).unsqueeze(0).to(device)
inputframe = torch.from_numpy(input_image).unsqueeze(0).to(device)

net = VideoCompressor(True).to(device)
net.load_state_dict(
    torch.load('XXX.pth', map_location='cpu'),
    strict=False)
net.eval()

dec_start = time.time()
clipped_recon_image = net(refframe, string_, string_shape)

# distortion MSE
loss_fn = torch.nn.MSELoss()
mse_loss = loss_fn(clipped_recon_image, inputframe)
psnr = torch.mean(10 * (torch.log(1. / mse_loss) / np.log(10))).cpu().detach().numpy()
dec_time = time.time() - dec_start
print(
    f"{psnr:.4f} psnr |"
    f" Decoded in {dec_time:.2f}s"
)
