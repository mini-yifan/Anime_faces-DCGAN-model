import torch
import torch.nn as nn
from gan_model import Config, Generator
from matplotlib import pyplot as plt
import torchvision


def paint(img):
    '''显示图片'''
    img = img.detach().cpu()
    img = img.permute(1, 2, 0).numpy()
    plt.imshow(img)
    plt.show()

def generate_img(netG):
    netG.eval()
    with torch.no_grad():
        noise = torch.randn(1, Config.Gin_channel, 1, 1)
        img = netG(noise)
        img = img.reshape(-1, 64, 64)
        paint(img)
    return img

def main():
    config = Config
    config.ngf = 128
    netG = Generator(config)
    netG.load_state_dict(torch.load('netG.pt'))
    img = generate_img(netG)
    #存储img为png图像
    torchvision.utils.save_image(img, 'img.png')


if __name__ == '__main__':
    main()
