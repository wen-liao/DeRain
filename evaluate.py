from parameters import *
from preprocessing import *
import numpy as np
from matric import *

model = torch.load(MODEL_PATH).to(device)

test_dataset = DeRainDataset(is_test=True)
test_loader = DataLoader(test_dataset)

psnr = 0
psnr_ = 0
ssim = 0
ssim_ = 0
for i,(images,gts) in enumerate(test_loader):
    images,gts = torch_variable(images,True), torch_variable(gts,False)
    outputs = model(images.to(device))[-1]
    psnr += PSNR(outputs[0].cpu(),gts[0].cpu())
    ssim += SSIM(outputs[0].cpu(),gts[0].cpu())
    psnr_ += PSNR(images[0].cpu(),gts[0].cpu())
    ssim_ += SSIM(images[0].cpu(),gts[0].cpu())
psnr /= i+1
ssim /= i+1

print("PSNR: ",psnr)
print("SSIM: ",ssim)