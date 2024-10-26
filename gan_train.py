import torch
from torch import nn
from gan_data import GANdata, read_split_data
from gan_model import Config, Generator, Discriminator, weights_init
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt


writer = SummaryWriter("./logs") #tensorboard启动命令：在所在文件目录下 tensorboard --logdir logs

def paint(img):
    '''显示图片'''
    img = img.detach().cpu()
    img = img.permute(1, 2, 0).numpy()
    plt.imshow(img)
    plt.show()

def train_loop(netG, netD, train_loader, val_loader, criterion, optimizerG, optimizerD, epochs):
    for epoch in range(epochs):
        print("epoch: ", epoch+1)
        # 训练模式
        netG.train()
        netD.train()
        d_loss_sum_train = 0
        g_loss_sum_train = 0
        for train_img in train_loader:
            '''判别器的梯度更新'''
            real_img = train_img.to(Config.device)
            d_output = netD(real_img)
            # 真实图片损失
            real_loss = criterion(d_output, torch.ones_like(d_output))   #全1
            #高斯分布输入
            noise_1 = torch.randn(Config.batch_size, Config.Gin_channel, 1, 1, device=Config.device)
            # 生成图片
            g_output = netG(noise_1)
            #判别器判别生成的图片
            d_output = netD(g_output.detach())
            # 假图片损失
            fake_loss = criterion(d_output, torch.zeros_like(d_output))  #全0
            #判别器总损失
            d_loss = real_loss + fake_loss
            d_loss_sum_train += d_loss.item()
            #清空梯度,迭代
            optimizerD.zero_grad()
            d_loss.backward()
            optimizerD.step()

            '''生成器的梯度更新'''
            noise_2 = torch.randn(Config.batch_size, Config.Gin_channel, 1, 1, device=Config.device)
            # 生成图片
            g_output = netG(noise_2)
            # 判别器判别生成的图片
            d_output = netD(g_output)
            # 生成图片损失
            g_loss = criterion(d_output, torch.ones_like(d_output))  #全1
            g_loss_sum_train += g_loss.item()
            # 清空梯度，迭代
            optimizerG.zero_grad()
            g_loss.backward()
            optimizerG.step()

        print("d_loss_train= ", d_loss_sum_train/len(train_loader))
        print("g_loss_train= ", g_loss_sum_train/len(train_loader))

        netD.eval()
        netG.eval()
        d_loss_sum_val = 0
        g_loss_sum_val = 0
        with torch.no_grad():
            for val_img in val_loader:
                '''判别器val'''
                real_img = val_img.to(Config.device)
                d_output = netD(real_img)
                # 真实图片损失
                real_loss = criterion(d_output, torch.ones_like(d_output))  # 全1
                # 高斯分布输入
                noise_1 = torch.randn(Config.batch_size, Config.Gin_channel, 1, 1, device=Config.device)
                # 生成图片
                g_output = netG(noise_1)
                # 判别器判别生成的图片
                d_output = netD(g_output)
                # 假图片损失
                fake_loss = criterion(d_output, torch.zeros_like(d_output))  # 全0
                # 判别器总损失
                d_loss = real_loss + fake_loss
                d_loss_sum_val += d_loss.item()

                '''生成器val'''
                noise_2 = torch.randn(Config.batch_size, Config.Gin_channel, 1, 1, device=Config.device)
                # 生成图片
                g_output = netG(noise_2)
                # 判别器判别生成的图片
                d_output = netD(g_output)
                # 生成图片损失
                g_loss = criterion(d_output, torch.ones_like(d_output))  # 全1
                g_loss_sum_val += g_loss.item()

            print("d_loss_val= ", d_loss_sum_val/len(val_loader))
            print("g_loss_val= ", g_loss_sum_val/len(val_loader))

            #写入tensorboard
            loss_d_g = {"d_loss_val": d_loss_sum_val/len(val_loader), "g_loss_val": g_loss_sum_val/len(val_loader),
                        "d_loss_train": d_loss_sum_train/len(train_loader), "g_loss_train": g_loss_sum_train/len(train_loader)}
            writer.add_scalars("loss", loss_d_g, epoch)

        #绘画
        noise = torch.randn(1, Config.Gin_channel, 1, 1, device=Config.device)
        img = netG(noise)
        img = img.reshape(-1, 64, 64)
        paint(img)

def main():
    #参数，模型
    config = Config()
    device = config.device
    netG = Generator(config).to(device)
    netD = Discriminator(config).to(device)
    netG.load_state_dict(torch.load('netG (3).pt'))
    netD.load_state_dict(torch.load('netD (3).pt'))

    # 网络初始化
    #netG.apply(weights_init)
    #netD.apply(weights_init)

    #数据加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((64, 64)),
        #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_path, val_path = read_split_data(".\\anime-faces")
    train_data = GANdata(train_path, transform)
    val_data = GANdata(val_path, transform)

    #数据迭代器
    batch_size = config.batch_size
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=3)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=3)

    #数据优化器
    lr = config.lr
    optimizerG = torch.optim.Adam(netG.parameters(), lr, betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(netD.parameters(), lr, betas=(0.5, 0.999))

    #损失函数
    criterion = nn.BCELoss()

    #迭代次数
    epochs = config.epochs
    train_loop(netG, netD, train_loader, val_loader, criterion, optimizerG, optimizerD, epochs)

    #保存模型
    netG.cpu()
    netD.cpu()
    torch.save(netG.state_dict(), ".\\netG (3).pt")
    torch.save(netD.state_dict(), ".\\netD (3).pt")

if __name__ == '__main__':
    main()

