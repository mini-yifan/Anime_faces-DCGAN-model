import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np

def read_split_data(root, val_rate=0.1, shuffle=True):
    img_list = [img for img in os.listdir(root) if img.endswith(".png")]# 获取所有图片路径
    if shuffle:
        random.shuffle(img_list)

    num_train = int(len(img_list)*(1-val_rate))
    img_list = [os.path.join(root, img) for img in img_list]

    train_path = img_list[:num_train]
    val_path = img_list[num_train:]
    return train_path, val_path


class GANdata(Dataset):
    def __init__(self, img_path, transform=transforms.ToTensor()):
        self.img_path = img_path
        self.transform = transform

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, item):
        img = Image.open(self.img_path[item])
        # 图片数据预处理
        img = self.transform(img)
        return img


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((64, 64)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_path, val_path = read_split_data(".\\anime-faces")
    train_data = GANdata(train_path, transform)
    val_data = GANdata(val_path, transform)

    print(len(train_data), train_data[0].shape)
    print(len(val_data))

    max = torch.max(train_data[1])
    min = torch.min(train_data[1])
    print(max, min)
    #print(train_data[0])

    array_rgb = train_data[0].permute(1, 2, 0).numpy()  # 转置并转换为 NumPy 数组

    plt.imshow(array_rgb)
    plt.show()
