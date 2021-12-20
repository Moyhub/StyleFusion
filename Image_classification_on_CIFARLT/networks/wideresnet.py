
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class StyleBlock(nn.Module):
    def __init__(self,feature,style_type,alpha1,alpha2):
        super(StyleBlock,self).__init__()
        self.features = feature 
        norm_layer = Adain2d

        self.norm1 = norm_layer(alpha1=alpha1,alpha2=alpha2)
        self.style_type = style_type

    def forward(self,content,labels):
        # 针对该数据集可以这样做
        if "Inclass" in self.style_type:
            # print("USING INCLASS")
            In_style = torch.zeros_like(content)
            unique_labels = labels.unique()
            for label in unique_labels:
                label_index = torch.where(labels==label)[0]
                if len(label_index) > 1:
                    index = torch.randint( high=len(label_index)-1 , size=(1,))
                else:
                    index = 0
                style_index = label_index[index]
                In_style[label_index] = content[style_index,:,:,:]

        if "Outclass" in self.style_type:
            # print("USING OUTCLASS")
            out_style = torch.zeros_like(content)
            unique_labels = labels.unique()
            for label in unique_labels:
                label_index_inclass = torch.where(labels==label)[0]         
                label_index_outclass = torch.where(labels!=label)[0]
                index = torch.randint( high=len(label_index_outclass)-1 , size=(1,))
                style_index = label_index_outclass[index]

                out_style[label_index_inclass] = content[style_index,:,:,:]
        else:
            raise Exception('The style_block must be Inclass or Outclass, Privided:')

        out = self.norm1(content,In_style,out_style)
        
        return out

class Adain2d(nn.Module):
    def __init__(self,eps=1e-5,momentum=0.1,alpha1=0,alpha2=0):
        super(Adain2d,self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.weight = True
        self.bias = None
        self.weight1 = alpha1
        self.weight2 = alpha2
    
    def forward(self,x,In_sytle,Out_style):
        b,c,h,w = x.size()

        # Inclass
        style_var = In_sytle.view(b,c,-1).var(dim=2) + self.eps
        style_std = style_var.sqrt().view(b,c,1,1)
        style_mean = In_sytle.view(b,c,-1).mean(dim=2).view(b,c,1,1)        

        # Outclass
        out_style_var = Out_style.view(b,c,-1).var(dim=2) + self.eps
        out_style_std =  out_style_var.sqrt().view(b,c,1,1)
        out_style_mean = Out_style.view(b,c,-1).mean(dim=2).view(b,c,1,1)    

        # 原始数据
        x_var = x.view(b,c,-1).var(dim=2) + self.eps
        x_std = x_var.sqrt().view(b,c,1,1)
        x_mean = x.view(b,c,-1).mean(dim=2).view(b,c,1,1)  

        norm_feat = (x - x_mean.expand(x.size()))/x_std.expand(x.size())
        if self.weight:
            style_std = (1-self.weight1-self.weight2) * x_std + self.weight1 * style_std + self.weight2 * out_style_std 
            style_mean = (1-self.weight1-self.weight2) * x_mean + self.weight1 * style_mean + self.weight2 * out_style_mean

        return norm_feat*style_std.expand(x.size()) + style_mean.expand(x.size())


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.3):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.3):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.3,add_style=False, styleratio1=0,styleratio2=0, style_type=["Inclass"],style_position=[0],seed=818):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        self.threthod = 0.2
        self.add_style = add_style
        self.style_position = style_position

        self.style_block = StyleBlock(feature=64,style_type=style_type,alpha1=styleratio1,alpha2=styleratio2)
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        self.feature_num = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x,label=None):
        out = self.conv1(x)

        if 0 in self.style_position and self.add_style and label!=None and random.random()<self.threthod:
            out = self.style_block(out,label)
        out = self.block1(out)
        if 1 in self.style_position and self.add_style and label!=None and random.random()<self.threthod:
            out = self.style_block(out,label)
        out = self.block2(out)
        if 2 in self.style_position and self.add_style and label!=None and random.random()<self.threthod:
            out = self.style_block(out,label)
        out = self.block3(out)
        if 3 in self.style_position and self.add_style and label!=None and random.random()<self.threthod:
            out = self.style_block(out,label)

        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        out = self.fc(out)
        return out #, self.fc(out)
