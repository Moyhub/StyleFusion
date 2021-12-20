import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
import torch
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


class ResNeXtBottleneck(nn.Module):
  expansion = 4
  """
  RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
  """
  def __init__(self, inplanes, planes, cardinality, base_width, stride=1, downsample=None):
    super(ResNeXtBottleneck, self).__init__()

    D = int(math.floor(planes * (base_width/64.0)))
    C = cardinality

    self.conv_reduce = nn.Conv2d(inplanes, D*C, kernel_size=1, stride=1, padding=0, bias=False)
    self.bn_reduce = nn.BatchNorm2d(D*C)

    self.conv_conv = nn.Conv2d(D*C, D*C, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
    self.bn = nn.BatchNorm2d(D*C)

    self.conv_expand = nn.Conv2d(D*C, planes*4, kernel_size=1, stride=1, padding=0, bias=False)
    self.bn_expand = nn.BatchNorm2d(planes*4)

    self.downsample = downsample

  def forward(self, x):
    residual = x

    bottleneck = self.conv_reduce(x)
    bottleneck = F.relu(self.bn_reduce(bottleneck), inplace=True)

    bottleneck = self.conv_conv(bottleneck)
    bottleneck = F.relu(self.bn(bottleneck), inplace=True)

    bottleneck = self.conv_expand(bottleneck)
    bottleneck = self.bn_expand(bottleneck)

    if self.downsample is not None:
      residual = self.downsample(x)
    
    return F.relu(residual + bottleneck, inplace=True)

class CifarResNeXt(nn.Module):
  """
  ResNext optimized for the Cifar dataset, as specified in
  https://arxiv.org/pdf/1611.05431.pdf
  """
  def __init__(self, block, depth, cardinality, base_width, num_classes,add_style=False, styleratio1=0,styleratio2=0, style_type=["Inclass"],style_position=[0],dropout_rate=0.1,seed=818):
    super(CifarResNeXt, self).__init__()

    #Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
    assert (depth - 2) % 9 == 0, 'depth should be one of 29, 38, 47, 56, 101'
    layer_blocks = (depth - 2) // 9

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    self.threthod = 0.2
    self.add_style = add_style
    self.style_position = style_position
    self.style_block = StyleBlock(feature=64,style_type=style_type,alpha1=styleratio1,alpha2=styleratio2)

    self.cardinality = cardinality
    self.base_width = base_width
    self.num_classes = num_classes

    self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    self.bn_1 = nn.BatchNorm2d(64)

    self.inplanes = 64
    self.stage_1 = self._make_layer(block, 64 , layer_blocks, 1)
    self.stage_2 = self._make_layer(block, 128, layer_blocks, 2)
    self.stage_3 = self._make_layer(block, 256, layer_blocks, 2)
    self.avgpool = nn.AvgPool2d(8)

    self.feature_num = 256*block.expansion
    self.classifier = nn.Linear(256*block.expansion, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        init.kaiming_normal(m.weight)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
              kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, self.cardinality, self.base_width, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes, self.cardinality, self.base_width))

    return nn.Sequential(*layers)

  def forward(self, x,label=None):
    x = self.conv_1_3x3(x)
    x = F.relu(self.bn_1(x), inplace=True)
    if 0 in self.style_position and self.add_style and label!=None and random.random()<self.threthod:
      print("888****")
      x = self.style_block(x,label)
    x = self.stage_1(x)
    if 1 in self.style_position and self.add_style and label!=None and random.random()<self.threthod:
      x = self.style_block(x,label)
    x = self.stage_2(x)
    if 2 in self.style_position and self.add_style and label!=None and random.random()<self.threthod:
      x = self.style_block(x,label)
    x = self.stage_3(x)
    if 3 in self.style_position and self.add_style and label!=None and random.random()<self.threthod:
      x = self.style_block(x,label)
      
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)

    x = self.classifier(x)
    return x

def resnext29_16_64(num_classes=10,add_style=False,styleratio1=0.2,styleratio2=0.2,style_type="Inclass",style_position=[0],dropout_rate=0.1,seed=818):
  """Constructs a ResNeXt-29, 16*64d model for CIFAR-10 (by default)
  
  Args:
    num_classes (uint): number of classes
  """
  model = CifarResNeXt(ResNeXtBottleneck, 29, 16, 64, num_classes)
  return model

def resnext29_8_64(num_classes=10,add_style=False,styleratio1=0.2,styleratio2=0.2,style_type="Inclass",style_position=[0],dropout_rate=0.1,seed=818):
  """Constructs a ResNeXt-29, 8*64d model for CIFAR-10 (by default)
  
  Args:
    num_classes (uint): number of classes
  """
  model = CifarResNeXt(ResNeXtBottleneck, 29, 8, 64, num_classes,add_style,styleratio1,styleratio2,style_type,style_position,dropout_rate,seed)
  return model