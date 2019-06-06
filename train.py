#data preprocessing
from torch.utils.data import Dataset,DataLoader
import numpy as np
from models import *
from preprocessing import *
from parameters import *
from loss import *

#processing data
train_dataset = DeRainDataset()
eval_dataset = DeRainDataset(is_eval=True)
test_dataset = DeRainDataset(is_test=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SZ, shuffle=True)
eval_loader = DataLoader(eval_dataset)

netG = Generator().to(device)
netD = Discriminator().to(device)

#MODELG_PATH = './weights/netG_epoch_200_ID_1_LOSS_0.03'
#MODELD_PATH = './weights/netD_epoch_200_ID_1_LOSS_1.38'
#netG = torch.load(MODELG_PATH).to(device)
optimG = torch.optim.Adam(netG.parameters(),lr=LR,betas=(0.5,0.99))
#netD = torch.load(MODELD_PATH).to(device)
optimD = torch.optim.Adam(netD.parameters(),lr=LR,betas=(0.5,0.99))

#check the validity of the model

with torch.no_grad():
    input_img = torch.rand((1,3,224,224)).to(device)
    output = netG(input_img)
    masks,output_img = output[0], output[-1]
'''
    plot(output_img[0])
    for mask in masks:
        plot(mask_to_image(mask[0]))
'''


for epoch in range(0,1+EPOCHES):
    loss_D_t , loss_G_t = 0,0
    count = 0
    for i, (datas,gts) in enumerate(train_loader):
        count += 1
        if count == 20:
            print(count)
        masks = []
        for i in range(datas.shape[0]):
            masks.append(get_mask(datas[i],gts[i]))
        masks = np.array(masks)

        datas,masks,gts = torch_variable(datas,True), torch_variable(masks,False), torch_variable(gts,False)
        As, T1, T2, T3 = netG(datas)
        Ts = [T1,T2,T3]
        Os = T3
        #print("GENERATOR")
        #loss of generator: attention loss + perceptual loss + multiscale loss + GAN loss
        loss_G = ATTENTIVE_AUTOENCODER*AttentionLoss(As, masks)
        loss_G += PerceptualLoss(gts, T3)
        loss_G += MultiScaleLoss(Ts,gts.cpu(),batch=BATCH_SZ)
        
        D_map_O, D_fake = netD(Os)
        loss_G += LAMBDA*GANLoss(D_fake,True)

        #print("DISCRIMINATOR")
        #loss of discriminator: map loss + GANloss
        D_map_R, D_real = netD(gts)
        loss_D = GANLoss(D_fake,False) + GANLoss(D_real,True) + GAMMA*MAPLoss(D_map_O,D_map_R,As[-1].detach())

        optimG.zero_grad()
        loss_G.backward(retain_graph=True)
        optimG.step()
        
        optimD.zero_grad()
        loss_D.backward()
        optimD.step()

        loss_G_t += loss_G.item()
        loss_D_t += loss_D.item()

    loss_G_t, loss_D_t = loss_G_t/count, loss_D_t/count
    if epoch % 10 == 0:
        for i in range(Os.shape[0]):
            #print(datas[i],Os[i],mask_to_image(As[-1][i]),mask_to_image(masks[i]),gts[i])
            img = torch.cat((datas[i],Os[i],mask_to_image(As[-1][i]),mask_to_image(masks[i]),gts[i]),dim=1)
            save(img,'./images/epoch_'+str(epoch)+"_ID_"+ID+'_'+str(i)+'.png')
            
        print("EPOCH: ", epoch+200)
        print("LOSS G: ", loss_G_t)
        print("LOSS D: ", loss_D_t)
    if epoch in [20,40,60,80,100,120,140,160,180,200]:
        torch.save(netG,'./weights/netG_epoch_'+str(epoch)+"_ID_"+ID+"_LOSS_"+str(truncate(loss_G_t)))
        torch.save(netD,'./weights/netD_epoch_'+str(epoch)+"_ID_"+ID+"_LOSS_"+str(truncate(loss_D_t)))