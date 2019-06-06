import torch

DISCRIMINATOR = True
ATTENTIVE_DISCRIMINATOR = True
ATTENTIVE_AUTOENCODER = True

#parameters
TRAIN_PATH = '../dataset/train/'
EVAL_PATH = '../dataset/test_a/'
TEST_PATH = '../dataset/test_b/'
BATCH_SZ = 2
THETA = 0.6
ITERATION = 4
GAMMA = 0.05*ATTENTIVE_DISCRIMINATOR
EPOCHES = 200
LR = 0.0005
LAMBDA = 0.01*DISCRIMINATOR
ID = '4'
BETA = 0.6

MODEL_PATH='./weights/netG_epoch_200_ID_1_LOSS_0.03'

#device
device = torch.device('cpu')