import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

def getGaussPyr_root(img_root,n):
    l=[]
    img=Image.open(img_root)
    l.append(img)
    img=np.array(img)
    last=img
    for i in range(n):
        img2=cv2.pyrDown(last)
        img_pil = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        print(type(img_pil))
        l.append(img_pil)            #此处图片为np.array，因为还要经过transform，要转为PIL图片
        last=img2
    return l

def getGaussPyr_img(img,n):  #经过transforms后 img为tensor
    #输入一个tensor 图片(PIL转换)，返回一个tensor列表，代表该图片经过高斯金字塔后的结果
    l = []
    l.append(img)
    img=np.array(img)
    print(img.shape)
    last=img
    for i in range(n):
        img2=cv2.pyrDown(last)
        img_pil = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        #print(type(img_pil))
        img_tensor=transform_train(img_pil)
        l.append(img_pil)            #此处图片为np.array，因为还要经过transform，要转为PIL图片
        last=img2
    return l


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