import numpy as np
import math
import cv2

def psnr(img1, img2):

    img2 = img2.mul(255)
    img2 = img2.cpu().detach().numpy().squeeze(0).transpose((1, 2, 0))
    # img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

    mse = np.mean((img1 / 255.0 - img2 / 255.0) ** 2)
    if mse < 1e-10:
        return 100
    psnr = 20 * math.log10(1 / math.sqrt(mse))
    return psnr