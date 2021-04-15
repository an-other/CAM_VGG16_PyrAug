from PIL import Image
import numpy as np
import torch
import cv2
from torchvision import transforms

int_to_class={0:'dog',1:'cat'}

def getcam(net,img,int_to_class,output_name):   #1张image(PIL)通过transform后传入
    #img=Image.open(img_root)
    input=img
    img = transform(img)
    #input=img
    #cv2.imread('input',img)
    _, h, w = img.shape
    img=img.unsqueeze(0)
    y=net(img)    #(1,C)
    feature_map=net.feature_map  # (1,C,h,w)
    feature_map=torch.squeeze(feature_map)  #(c,h,w)
    print(feature_map.shape)

    y=torch.squeeze(y)

    prob,index=torch.sort(y,dim=0,descending=True)
    line="{:.3f}->{}".format(prob[0],int_to_class[index[0].item()])
    print(line)

    feature=feature_map[index[0].item()]
    #print(type(feature))
    img_cv2 = cv2.cvtColor(np.asarray(input),cv2.COLOR_RGB2BGR)
    #img_cv2=cv2.imread(img_root)
    h, w, _ = img_cv2.shape

    feature=feature-torch.min(feature)
    feature/=torch.max(feature)
    feature=np.uint8(255*feature.detach())
    #feature=cv2.resize(feature,(256,256))
    feature=cv2.resize(feature,(w,h))

    heatmap = cv2.applyColorMap(feature, cv2.COLORMAP_JET)
    #img=torch.squeeze(img).numpy().transpose(1,2,0)

    #heatmap=heatmap.transpose(2,0,1)
    print(heatmap.shape,img_cv2.shape)

    res=heatmap*0.4+img_cv2*0.6
    #print(res.shape)
    cv2.imwrite(output_name+'.jpg', res)
    print('cam done')
    return res



normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
)

transform = transforms.Compose([
    #transforms.RandomResizedCrop(224),
    #transforms.RandomHorizontalFlip(),
    transforms.Resize(224),
    transforms.ToTensor(),
    normalize
])

