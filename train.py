import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter
from utils import imshow, weights_init_normal
from collections import deque
from tqdm import tqdm
import itertools

from utils import compute_inception_score
from network import Generator, Discriminator

writer = SummaryWriter()

###########
# Parameter
###########
data = 'CIFAR10'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter('tensorlog/dcgan_'+data)
print(device)
epoch = 250
img_size = 32
bsize = 64
ngf = 64
ndf = 64
nz = 128
lrG = 0.0004
lrD = 0.0002
iter = 0
netG = Generator(nz=nz, ngf=ngf, nc=3).to(device)
netD = Discriminator(ndf=ndf, nc=3).to(device)
netG.apply(weights_init_normal)
netD.apply(weights_init_normal)
criterion = nn.BCEWithLogitsLoss()
fixed_noise = torch.randn(64, nz, 1, 1).to(device)
optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas=(0.5, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lrD, betas=(0.5, 0.999))

ctrl = 2
best_is = 0
lossD = torch.tensor(0)
lossG = torch.tensor(0)
dTrue = deque(maxlen=1000)
dFalse = deque(maxlen=1000)
real_label = torch.FloatTensor(bsize).fill_(.9).to(device)
fake_label = torch.FloatTensor(bsize).fill_(.1).to(device)
use_test = False  # Set True to use sample from test set
#############
# Data loader
#############
print("load dataset...")
train_set = datasets.CIFAR10('./data', train=True, download=True,
                     transform=transforms.Compose([
                                transforms.Resize(img_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ]))
test_set = datasets.CIFAR10('./data', train=False, download=True,
                     transform=transforms.Compose([
                                transforms.Resize(img_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ]))
train_loader = torch.utils.data.DataLoader(
             train_set,
             batch_size=bsize, shuffle=True,
             num_workers=0, pin_memory=True,
             drop_last=True)

test_loader = torch.utils.data.DataLoader(
             test_set,
             batch_size=bsize, shuffle=True,
             num_workers=0, pin_memory=True,
             drop_last=True)

loader = itertools.chain(train_loader, test_loader) if use_test else train_loader

##########
# Training
##########
for e in range(epoch):
    for i, (x, y) in enumerate(loader):
        iter += 1
        noise = torch.randn(bsize, nz, 1, 1).to(device)
        x = x.to(device)
        # train Discriminator
        if i % ctrl == 0:
            optimizerD.zero_grad()

            outputTrue = netD(x)
            lossDT = criterion(outputTrue, real_label)

            outputFalse = netD(netG(noise).detach())
            lossDF = criterion(outputFalse, fake_label)

            lossD = lossDF + lossDT
            lossD.backward()
            optimizerD.step()

            dTrue.append(torch.sigmoid(outputTrue).data.cpu().numpy().mean())
            dFalse.append(torch.sigmoid(outputFalse).data.cpu().numpy().mean())
            writer.add_scalars('data/scalar', {"D(x)": np.array(dTrue).mean(), "D(G(x))": np.array(dFalse).mean()}, iter)

        # train Generator
        else:
            optimizerG.zero_grad()
            output = netD(netG(noise))
            lossG = criterion(output, real_label)
            lossG.backward()
            optimizerG.step()

    # compute inception score
    is_mean, is_std = compute_inception_score(netG, device)
    writer.add_scalars('data/inception', {"val": is_mean}, iter)

    # save images
    if e % 10 == 0:
        img = netG(fixed_noise).detach()
        writer.add_image('Image', vutils.make_grid(img.data, padding=2, normalize=True), e)

    # save network
    if is_mean > best_is:
        best_is = is_mean
        torch.save(netG.state_dict(), './network_save/best_dcgan_' + data + '_gen' + str(e) + 'epoch_'+str(round(is_mean, 2))+'.pth')
        torch.save(netD.state_dict(), './network_save/best_dcgan_' + data + '_dis' + str(e) + 'epoch_'+str(round(is_mean, 2))+'.pth')

    elif e % 50 == 0:
        torch.save(netG.state_dict(), './network_save/dcgan_' + data + '_gen' + str(e) + 'epoch.pth')
        torch.save(netD.state_dict(), './network_save/dcgan_' + data + '_dis' + str(e) + 'epoch.pth')

    print("Epoch:", e, "inception score:", round(is_mean, 4))

writer.close()
