import numpy as np
from skimage.measure import compare_psnr, compare_ssim
import cv2

def PSNR(output,gt):
    dif = np.array(output.detach()-gt.detach())
    return 10*np.log10(dif.size/np.sum(dif*dif))

def SSIM(im1, im2):
    im1,im2 = im1.detach().permute(1,2,0).numpy()*255, im2.detach().permute(1,2,0).numpy()*255
    im1,im2 = np.array(im1,dtype=np.uint8),np.array(im2,dtype=np.uint8)
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    return compare_ssim(im1_y, im2_y)
