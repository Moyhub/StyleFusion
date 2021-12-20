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
from networks.resnet import resnet110_cifar, resnet32_cifar
from networks.wideresnet import WideResNet
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# plt.switch_backend("agg")
def cal_In_Outclass_std(x,label,unique):
    mean_all = np.mean(x,axis=0)
    variance_all = 0
    intra_variance = 0
    for c in unique:
        classx = x[label==c]
        meanx = np.mean(classx,axis=0)
        variancex = (np.sum((classx[:,0]-meanx[0])**2 + (classx[:,1]-meanx[1])**2))/classx.shape[0]
        # print("class",c)
        # print("the variance:",variancex)
        intra_variance = intra_variance + variancex

        variance_all = variance_all + ((meanx[0]-mean_all[0])**2 + (meanx[1]-mean_all[1])**2 )*classx.shape[0]

    print("inter_class variance:",variance_all/label.shape[0])
    print("intra_class variance:",intra_variance/10)

def cal_In_Outclass_std_1d(x,label,unique):
    mean_all = np.mean(x,axis=0)
    variance_all = 0
    intra_variance = 0
    for c in unique:
        classx = x[label==c]
        meanx = np.mean(classx,axis=0)
        variancex = (np.sum((classx-meanx)**2))/classx.shape[0]
        # print("class",c)
        # print("the variance:",variancex)
        intra_variance = intra_variance + variancex

        variance_all = variance_all + ((meanx-mean_all)**2)*classx.shape[0]

    print("inter_class variance:",variance_all/label.shape[0])
    print("intra_class variance:",intra_variance/10)



def main(state_dicts):

    parser = argparse.ArgumentParser()
    parser.add_argument("--method",type=str,default="tsne",help="tsne or add the pca")
    args = parser.parse_args(args=[])
    
    model= resnet110_cifar()
    model = WideResNet(16,10,8,0)
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(state_dicts["state_dict"])

    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    lastmodel = model

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=transform),
        batch_size=128, shuffle=False)

    for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            lastmodel = lastmodel.cuda()

            # compute output
            output = lastmodel(input_var,None)
            output = output.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
            if i == 0:
                out = output
                tar = target
            else:
                out = np.concatenate((out,output),axis=0)
                tar = np.concatenate((tar,target),axis=0)
    out = out.reshape(out.shape[0],-1)

    if args.method == "tsne":
            tsne = TSNE(
                n_components=2, metric="cosine", verbose=1,
                perplexity=50, n_iter=1000, learning_rate=200.
            )
            embed2d = tsne.fit_transform(out)
    color = set(tar.tolist())
    # cal_In_Outclass_std(embed2d,tar,color)

    # print(max(embed2d[:,0]),min(embed2d[:,0]))
    # print(max(embed2d[:,1]),min(embed2d[:,1]))

    # for c in color:
    #     e = embed2d[tar==c]
    #     plt.xlim(-70,70)
    #     plt.ylim(-75,75)
    #     plt.scatter(e[:,0],e[:,1],s=2,label=1)
    #     plt.axis('off') 
    # plt.savefig("resnet110-origin.pdf")
    
    pca = PCA(n_components=2).fit_transform(out)
    cal_In_Outclass_std(pca,tar,color)
    # embed1d = PCA(n_components=1).fit_transform(out)
    # cal_In_Outclass_std_1d(embed1d,tar,color)

    return embed2d,color,tar

if __name__ == "__main__":
    # file_dir = ["ISDA_test/cifar10_resnet-110_/no_1_lambda_0_0.5_standard-Aug_nostyle_seed=196_/checkpoint/checkpoint.pth.tar","ISDA_test/cifar10_resnet-110_/no_1_lambda_0_0.5_standard-Aug__addstyle__styletype=InOutclass__styleratio1=0.2_styleratio2=0.4__seed=196__layer=[0, 1]/checkpoint/checkpoint.pth.tar"]
    
    file_dir = ["ISDA_test/cifar10_wideresnet-16-8_/no_1_lambda_0_0.5_standard-Aug__dropout_nostyle_seed=324_/checkpoint/checkpoint.pth.tar",
    "ISDA_test/cifar10_wideresnet-16-8_/no_1_lambda_0_0.5_standard-Aug__dropout__addstyle__styletype=InOutclass__styleratio1=0.2_styleratio2=0.4__seed=676__layer=[0, 1]/checkpoint/checkpoint.pth.tar"]
    embed2d_list = []

    for file in file_dir:
        state_dicts = torch.load(file)
        embed2d,color,tar = main(state_dicts)
        embed2d_list.append(embed2d)
    
    x_min = min( min(embed2d_list[0][:,0]),min(embed2d_list[1][:,0]) )
    x_max = max( max(embed2d_list[0][:,0]),max(embed2d_list[1][:,0]) )
    y_min = min( min(embed2d_list[0][:,1]),min(embed2d_list[1][:,1]) )
    y_max = max( max(embed2d_list[0][:,1]),max(embed2d_list[1][:,1]) )

    for i in range(len(embed2d_list)):
        embed2d = embed2d_list[i]
        for c in color:
            e = embed2d[tar==c]
            plt.xlim(x_min,x_max)
            plt.ylim(y_min,y_max)
            plt.scatter(e[:,0],e[:,1],s=2,label=1)
            plt.axis('off') 
            plt.savefig("wideresnet-origin"+str(i)+".jpg")
        plt.close()

