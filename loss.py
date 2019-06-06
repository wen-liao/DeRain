import torch
from torch import nn
from torchvision.models.vgg import vgg16
from models import *
from parameters import *

BCELoss = nn.BCELoss().to(device)
MSELoss = nn.MSELoss().to(device)

def GANLoss(input_, is_real):
    return BCELoss(input_,torch.FloatTensor(input_.size()).fill_(is_real*1.0).to(device))

def AttentionLoss(A_,M_):
        loss_ATT = 0
        for i in range(1, ITERATION+1):
                loss_ATT += pow(THETA, ITERATION-i) * BCELoss(A_[i-1],M_)
        return loss_ATT

vgg_model = vgg(vgg_init())

def PerceptualLoss(O_,T_):
    O = vgg_model(O_)
    T = vgg_model(T_)
    loss = 0
    for i in range(len(T)):
        loss += MSELoss(O[i],T[i])/float(len(T))
    return loss

def MultiScaleLoss(S_,gt,batch=1):
    gt = gt.permute((0,2,3,1))
    lambdas = [0.6,0.8,1.0]
    #ground truths
    T_ = []
    #resize gt by 1/4,1/2,1
    for i in range(S_[0].shape[0]):
        temp = []
        x = (np.array(gt[i])*255.).astype(np.uint8)
        t = cv2.resize(x, None, fx=1.0/4.0,fy=1.0/4.0, interpolation=cv2.INTER_AREA)
        t = np.expand_dims((t/255.).astype(np.float32).transpose(2,0,1),axis=0)
        temp.append(t)
        t = cv2.resize(x, None, fx=1.0/2.0,fy=1.0/2.0, interpolation=cv2.INTER_AREA)
        t = np.expand_dims((t/255.).astype(np.float32).transpose(2,0,1),axis=0)
        temp.append(t)
        x = np.expand_dims((x/255.).astype(np.float32).transpose(2,0,1),axis=0)
        temp.append(x)
        T_.append(temp)
    temp_T = []
    for i in range(len(lambdas)):
        for j in range((S_[0].shape[0])):
            if j == 0:
                x = T_[j][i]
            else:
                #print(x.shape,T_[j][i].shape)
                x = np.concatenate((x, T_[j][i]), axis=0)
                #print(x.shape)
        temp_T.append(torch.from_numpy(x).to(device))
    T_ = temp_T
    loss_ML = 0
    for i in range(len(lambdas)):
        #print(S_[i].shape,T_[i].shape)
        loss_ML += lambdas[i] * MSELoss(S_[i], T_[i])
    return loss_ML/float(S_[0].shape[0])

def MAPLoss(D_O,D_R,A_N):
    return MSELoss(D_O,A_N) + MSELoss(D_R,torch.zeros(D_R.shape).to(device))