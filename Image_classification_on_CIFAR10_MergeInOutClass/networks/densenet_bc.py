
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
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

def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                        kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        small_inputs (bool) - set to True if images are 32x32. Otherwise assumes images are larger.
        efficient (bool) - set to True to use checkpointing. Much more memory efficient, but slower.
    """
    def __init__(self, numclass=10,growth_rate=12, block_config=(16, 16, 16), compression=0.5,
                 num_init_features=24, bn_size=4, drop_rate=0,
                small_inputs=True, efficient=False,add_style=False, styleratio1=0,styleratio2=0, style_type=["Inclass"],style_position=[0],seed=818):

        super(DenseNet, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
        self.avgpool_size = 8 if small_inputs else 7

        # First convolution
        if small_inputs:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ]))
            self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
            self.features.add_module('relu0', nn.ReLU(inplace=True))
            self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                                           ceil_mode=False))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        self.threthod = 0.2
        self.add_style = add_style
        self.style_position = style_position

        self.style_block = StyleBlock(feature=64,style_type=style_type,alpha1=styleratio1,alpha2=styleratio2)

        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))

        # Linear layer

        self.feature_num = num_features
        self.classifier = nn.Linear(num_features, numclass) 

        # Initialization
        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                n = param.size(0) * param.size(2) * param.size(3)
                param.data.normal_().mul_(math.sqrt(2. / n))
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)
            elif 'classifier' in name and 'bias' in name:
                param.data.fill_(0)

    def forward(self, x,label=None):
        # features = self.features(x)
        for index,layer in enumerate(list(self.features)):
            x = layer(x)
            if index in self.style_position and self.add_style and label!=None and random.random()<self.threthod:
                x = self.style_block(x,label)
        features = x
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=self.avgpool_size).view(features.size(0), -1)
        out = self.classifier(out)
        return out
