# -*- coding: utf-8 -*-
import argparse
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from networks.resnet import resnet110_cifar,resnet32_cifar
from networks.wideresnet import WideResNet
import os
from autoaugment import CIFAR10Policy
from cutout import Cutout
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import copy
import torch.nn.functional as F
plt.rcParams["savefig.dpi"] = 300 # 图片像素
plt.rcParams["figure.dpi"] = 600 # 分辨率

parser = argparse.ArgumentParser(description='Using the Style Transfer on Image Classification Task')

parser.add_argument('--augment', type=bool, default=False, help="Whether to use stardard augmentation (default True)")
parser.set_defaults(augment=True) # 设置参数默认值
# Autoaugment
parser.add_argument('--autoaugment',type=bool,default=True,help='whether to use autoaugment')
# cutout
parser.add_argument('--cutout',type=bool,default=True,help='whether to use cutout')
parser.add_argument('--n_holes', type=int, default=1,help='number of holes to cut out from image')
parser.add_argument('--length', type=int, default=16,help='length of the holes')
parser.add_argument('--style', type=bool, default=True)
args = parser.parse_args()

# 计算神经网络的计算值
state_dicts = torch.load("Standardcheckpoint.pth.tar")

model= resnet110_cifar()
model_style = resnet110_cifar(add_style=True,styleratio1=0.5,styleratio2=0.2,style_type=["Inclass","Outclass"],style_position=[0,1],dropout_rate=0,seed=169)


model = torch.nn.DataParallel(model).cuda()
model.load_state_dict(state_dicts["state_dict"])

model_style = torch.nn.DataParallel(model_style).cuda()
model_style.load_state_dict(state_dicts["state_dict"])

normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
if args.augment:
    if args.autoaugment:
        print('Autoaugment')
        transform_train_AA = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                                (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(), CIFAR10Policy(),
            transforms.ToTensor(),
#             Cutout(n_holes=args.n_holes, length=args.length),
            normalize,
        ])

    if args.cutout:
        print('Cutout')
        transform_train_Cut = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                                (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            Cutout(n_holes=args.n_holes, length=args.length),
            normalize,
        ])

    if True:
        print('Standrad Augmentation!')
        transform_train = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])


    transform_normal = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
else:
    # transform_train = transforms.Compose([
    #             transforms.ToTensor(),
    #             normalize,
    #             ])
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=True, transform=transform_train),
    batch_size=256, shuffle=False)

def Getoutput(model,train_loader):
    count=0
    new_count = 0
    for trans in [transform_normal,transform_train_AA,transform_train_Cut,transform_train]:
        new_count+=1
        print(new_count)
        train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, transform=trans),
        batch_size=256, shuffle=False)


        for i, (input, target) in enumerate(train_loader):
            target = target.cuda()
            input_var = input.cuda()
            model = model.cuda()
            
            if args.style and new_count==4:
                output = model_style(input_var,target)
            else:
                output = model(input_var,None)
            output = output.detach().cpu().numpy()
            target = target.detach().cpu().numpy()

            # 转化为一个Numpy数组
            if i == 0:
                out = output
                tar = target
            elif i==40:
                break
            else:
                out = np.concatenate((out,output),axis=0)
                tar = np.concatenate((tar,target),axis=0)
        if count==0:
            newout = out
            count=+1
        else:
            newout = np.concatenate((newout,out),axis=0)
    return newout,tar

def visual(origindata,tar,splitpos):
    color = set(tar.tolist())   
    data1 = origindata[0:splitpos]
    for c in color:
        data = data1[tar==c]
        plt.scatter(data[:,0],data[:,1],s=3,marker=".")#c=tar,cmap=plt.cm.Spectral)
    plt.axis('off') 
#     plt.savefig("resnet110add"+str(0)+".jpg")
    plt.savefig("origin.pdf")
    plt.close()

#   AA
    data1 = origindata[splitpos:2*splitpos]
    for c in color:
        data = data1[tar==c]
        plt.scatter(data[:,0],data[:,1],s=3,marker="^")#,c=tar,marker=".",cmap=plt.cm.Spectral)
    
    data1 = origindata[0:splitpos]
    for c in color:
        data = data1[tar==c]
        plt.scatter(data[:,0],data[:,1],s=3,marker=".")#,alpha=0.4)#,c=tar,marker=".",cmap=plt.cm.Spectral)
    
    plt.axis('off') 
    plt.savefig("AA.pdf")  
    plt.close()

##  Cut
    data1 = origindata[2*splitpos:3*splitpos]
    for c in color:
        data = data1[tar==c]
        plt.scatter(data[:,0],data[:,1],s=3,marker="^")#,c=tar,marker=".",cmap=plt.cm.Spectral)
    
    data1 = origindata[0:splitpos]
    for c in color:
        data = data1[tar==c]
        plt.scatter(data[:,0],data[:,1],s=3,marker=".")#,alpha=0.4)#,c=tar,marker=".",alpha=0.3,cmap=plt.cm.Spectral)
    
    plt.axis('off') 
    plt.savefig("Cut.pdf")  
    plt.close()

##  Style

    data1 = origindata[3*splitpos:]
    for c in color:
        data = data1[tar==c]
        plt.scatter(data[:,0],data[:,1],s=3,marker="^")#,c=tar,marker=".",cmap=plt.cm.Spectral)
    
    data1 = origindata[0:splitpos]
    for c in color:
        data = data1[tar==c]
        plt.scatter(data[:,0],data[:,1],s=3,marker=".")#,alpha=0.4)#,c=tar,marker=".",alpha=0.3,cmap=plt.cm.Spectral)
    
    plt.axis('off') 
    plt.savefig("Style.pdf")  
    plt.close()


if __name__ == '__main__':
    out,tar = Getoutput(model,train_loader)
    splitpos = len(tar)

    tsne = TSNE(
        n_components=2,init="pca",metric="euclidean", verbose=0,
        perplexity=50, n_iter=1000, learning_rate=200.
    )
    out = tsne.fit_transform(out)

    print(splitpos)
    # 先画原始图像
    visual(out,tar,splitpos)
    
    # 再画增强图像
#     visual(out,tar,splitpos)
