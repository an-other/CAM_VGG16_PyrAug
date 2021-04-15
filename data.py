from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import torch
import random
import os
import numpy as np
import cv2

DIR_TRAIN = "E:/CAM/train/"
DIR_TEST = "E:/CAM/test/"
class_to_int={'dog':0,'cat':1}
int_to_class={0:'dog',1:'cat'}



class catdogDataset(Dataset):
    def __init__(self, imgs, class_to_int, transform=None):
        super(Dataset, self).__init__()

        self.imgs = imgs

        self.class_to_int = class_to_int
        self.transform = transform

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img = Image.open(DIR_TRAIN + img_name)
        img = self.transform(img)

        label = class_to_int[img_name.split('.')[0]]
        label = torch.tensor(label, dtype=torch.long)
        return img, label

    def __len__(self):
        return len(self.imgs)


class catdogDataset_aug(Dataset):
    def __init__(self, imgs, class_to_int, transform=None, n=2):
        super(Dataset, self).__init__()
        self.n = n
        self.imgs = imgs
        self.class_to_int = class_to_int
        self.transform = transform

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img = Image.open(DIR_TRAIN + img_name)
        label = class_to_int[img_name.split('.')[0]]
        label = torch.tensor(label, dtype=torch.long)
        data_l = []
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        for i in range(self.n):
            if i == 0:
                img2 = img_cv
                last = img2
            else:
                img2 = cv2.pyrDown(last)
                last = img2
            img2=Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
            img2 = self.transform(img2)

            data_l.append(img2)
        return data_l, label

    def __len__(self):
        return len(self.imgs)

normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
)

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])


transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

def get_data(img_root,batch_size,shuffle=True):
    imgs = os.listdir(img_root)
    random.shuffle(imgs)
    data = catdogDataset(imgs[:int(len(imgs) * 0.9)], class_to_int, transform_train)
    # train_data = datasets.ImageFolder('kaggle/working/train/', transform=transform_train)
    dataloader = DataLoader(data, batch_size=batch_size,shuffle=True)
    return dataloader

def get_data_aug(img_root,batch_size,n=2,shuffle=True):
    imgs = os.listdir(img_root)
    random.shuffle(imgs)
    data = catdogDataset_aug(imgs[:int(len(imgs) * 0.9)], class_to_int, transform_train,n=n)
    # train_data = datasets.ImageFolder('kaggle/working/train/', transform=transform_train)
    dataloader = DataLoader(data, batch_size=batch_size,shuffle=True)
    return dataloader