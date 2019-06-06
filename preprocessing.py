from torch.utils.data import Dataset,DataLoader
import glob
import cv2
import numpy as np
import torch
from parameters import *
import matplotlib.pyplot as plt

class DeRainDataset(Dataset):
    def __init__(self, is_eval=False, is_test=False):
        super(DeRainDataset, self).__init__()

        if is_test:
            self.dataset = TEST_PATH
        elif is_eval:
            self.dataset = EVAL_PATH
        else:
            self.dataset = TRAIN_PATH

        self.img_list = sorted(glob.glob(self.dataset+'/data/*'))
        self.gt_list = sorted(glob.glob(self.dataset+'/gt/*'))
   
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        gt_name = self.gt_list[idx]

        img = cv2.imread(img_name,-1)
        gt = cv2.imread(gt_name,-1)
        #each image is resized into a 3*224*224 numpy array and each pixel is squeezed into an interval of [0,1)
        img = cv2.resize(img, (224,224), interpolation=cv2.INTER_AREA)
        gt = cv2.resize(gt, (224,224), interpolation=cv2.INTER_AREA)

        if img.dtype == np.uint8:
            img = (img / 255.0).astype('float32')
        if gt.dtype == np.uint8:
            gt = (gt / 255.0).astype('float32')

        return [img,gt]

def get_mask(dg_img,img):
    #convert to numpy arrays
    dg_img, img = np.array(dg_img), np.array(img)

	# downgraded image - image
    mask = np.fabs(dg_img-img)
	
    # threshold under 30
    mask[np.where(mask<(30.0/255.0))] = 0.0
    mask[np.where(mask>0.0)] = 1.0
    #avg? max?
    mask = np.max(mask, axis=2)
    mask = np.expand_dims(mask, axis=2)
    return mask


def torch_variable(x, is_train):
	return torch.tensor(np.array(x).transpose((0,3,1,2)),requires_grad=is_train).to(device)

#input is a 3*224*224 tensor
def plot(x):
    img = x.detach().cpu().numpy().transpose(1,2,0)
    plt.imshow(img)
    plt.show()

#input is a 1*224*224 tensor
def mask_to_image(x):
    return torch.cat((x,x,x),dim=0)

def truncate(x):
    return int(100*x)/100
    
