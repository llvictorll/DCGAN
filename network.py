import torch.nn as nn

class Discriminator(nn.Module):
    """Discriminator (DCGAN) for image size 32x32"""
    def __init__(self, ndf, nc=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(nc, ndf * 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf * 1, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=1, padding=0, bias=False)
        )

    def forward(self, input):
        return self.model(input).squeeze()


class Generator(nn.Module):
    """Generator (DCGAN) for image size 32x32"""
    def __init__(self, nz, ngf, nc=3):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(ngf * 2, ngf * 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 1),
            nn.ReLU(),
            nn.ConvTranspose2d(ngf * 1, nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.model(input)
