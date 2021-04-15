"""
Class Activation Mapping
Googlenet, Kaggle data
"""


from train import *
import torch, os
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets, transforms
import torchvision
from PIL import Image
from vgg16_bn import *
from get_cam import *
from data import *
from utils import getGaussPyr_root,getGaussPyr_img

import random



#build datasets
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
        label = class_to_int[img_name.split('.')[0]]
        label = torch.tensor(label, dtype=torch.long)
        img=self.transform(self)
        return img,label

    def __len__(self):
        return len(self.imgs)


class catdogDataset_aug(Dataset):
    def __init__(self,imgs,class_to_int,transform=None,n=2):
        super(Dataset,self).__init__()
        self.n=n
        self.imgs=imgs
        self.class_to_int=class_to_int
        self.transform=transform

    def __getitem__(self,idx):
        img_name=self.imgs[idx]
        img=Image.open(DIR_TRAIN+img_name)
        label = class_to_int[img_name.split('.')[0]]
        label = torch.tensor(label, dtype=torch.long)
        data_l=[]
        img_cv=cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR)
        for i in range(self.n):
            if i==0:
                img2=img_cv
                last=img2
            else:
                img2=cv2.pyrDown(last)
                last=img2
            img2=self.transform(img2)
            
            data_l.append(img2)
        return data_l,label
    	
    def __len__(self):
        return len(self.imgs)


# prepare data
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
transform = transforms.Compose([
    #transforms.RandomResizedCrop(224),
    #transforms.RandomHorizontalFlip(),
    transforms.Resize(224),
    transforms.ToTensor(),
    normalize
])
transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])


# class
classes = {0: 'dog', 1: 'cat'}




#net.cuda()

# functions
CAM             = 1
USE_CUDA        = 0
RESUME          = 0
PRETRAINED      = 1
GET_CAM         = 0
TRAIN           = 1
Guass_test      = 0
TEST            = 0

# hyperparameters
BATCH_SIZE      = 32
IMG_SIZE        = 224
LEARNING_RATE   = 0.00001
EPOCH           = 2





if __name__ == '__main__':
    net=get_model(2,pretrained=True)

    if USE_CUDA:
        net=net.cuda()
    if GET_CAM:
        net.eval()
        img_root="cat.32.jpg"
        img = Image.open(img_root)
        output_name=img_root[:5]+"cam"
        getcam(net,img,int_to_class,output_name)

    if Guass_test:
        net.eval()
        img_root="cat.37.jpg"
        gauss_list=getGaussPyr_root(img_root,n=2)

        for index,img in zip(range(len(gauss_list)),gauss_list):
            output_name=img_root[:5]+"cam"+str(index)
            getcam(net,img,int_to_class,output_name)
            #img.show()

    if TRAIN:
        trainloader=get_data_aug(DIR_TRAIN,BATCH_SIZE)
        testloader=get_data(DIR_TEST,BATCH_SIZE)
        criterion = torch.nn.CrossEntropyLoss()
        if PRETRAINED:
            optimizer = torch.optim.SGD(net.transf.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
        else:
            optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)

        for epoch in range(1, EPOCH + 1):
            retrain_aug(trainloader, net, USE_CUDA, epoch, criterion, optimizer)
            retest(testloader, net, USE_CUDA, criterion, epoch, RESUME)
    if TEST:
        img_root = "cat.4.jpg"
        img=Image.open(img_root)
        l=getGaussPyr_img(img,3)
        for i in l:
            i.show()