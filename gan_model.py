import torch
import torch.nn as nn
from torchsummary import summary
from torch.nn import functional as F

#初始化超参数
class Config:
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    dropout = 0.157754
    batch_size = 64
    lr = 0.0002
    Gin_channel = 100
    Gout_channel = 3
    ngf = 64
    Din_channel = 3
    Dout_channel = 1
    ndf = 64
    epochs = 20
    show_img_num = 20


#权重初始化
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    '''生成模型'''
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        self.in_channel = config.Gin_channel
        self.out_channel = config.Gout_channel
        self.ngf = config.ngf
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.in_channel, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(self.ngf, self.out_channel, 4, 2, 1, bias=False),
            #nn.Tanh()
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    '''判别模型'''
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config
        self.in_channel = config.Din_channel
        self.out_channel = config.Dout_channel
        self.ndf = config.ndf
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.in_channel, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8, self.out_channel, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


if __name__ == '__main__':
    config = Config
    device = config.device
    netG = Generator(config).to(config.device)
    netD = Discriminator(config).to(config.device)

    netG.apply(weights_init)
    netD.apply(weights_init)

    print(netG)
    print(netD)

    #查看模型结构
    summary(netG, (Config.Gin_channel, 1, 1))
    summary(netD, (Config.Din_channel, 64, 64))

    a = torch.randn(1, 100, 1, 1).to(device)
    output = netG(a)
    # 求四维张量中的最大值,最小值
    max_value = torch.max(output)
    min_value = torch.min(output)
    print(max_value, min_value)











