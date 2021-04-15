import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision

class VGG(nn.Module):
    def __init__(self,feature, num_classes):
        super().__init__()
        self.num_classes = num_classes
        # define an empty for Conv_ReLU_MaxPool
        net = []
        
        
        # add net into class property
        self.extract_feature = feature
        self.transf=nn.Conv2d(512,num_classes,1)
        #CAM_GAP


        """
        #ori vgg
        # define an empty container for Linear operations
        classifier = []
        classifier.append(nn.Linear(in_features=512*7*7, out_features=4096))
        classifier.append(nn.ReLU())
        classifier.append(nn.Dropout(p=0.5))
        classifier.append(nn.Linear(in_features=4096, out_features=4096))
        classifier.append(nn.ReLU())
        classifier.append(nn.Dropout(p=0.5))
        classifier.append(nn.Linear(in_features=4096, out_features=self.num_classes))
        
        # add classifier into class property
        self.classifier = nn.Sequential(*classifier)
        """

    def forward(self, x):
        feature = self.extract_feature(x)  #(B,num_classes,H,W)

        feature=self.transf(feature)
        self.feature_map = feature
        y = F.adaptive_avg_pool2d(feature,(1,1))
        y=y.view(y.shape[0],-1)
        y=F.softmax(y)
        return y


def get_model(num_class,pretrained=True):
    if pretrained:
        feature = torchvision.models.vgg16_bn(pretrained=False).features
        net = VGG(feature, num_class)
        net.load_state_dict(
            torch.load('E:/0github/vgg16_bn0.pth', map_location=torch.device('cpu')))  # input your pretrained net
        print('load model successful\n')
        for param in net.extract_feature.parameters():
            param.requires_grad = False

    else:
        feature = torchvision.models.vgg16_bn(pretrained=False).features
        net = VGG(feature, 2)

    return net