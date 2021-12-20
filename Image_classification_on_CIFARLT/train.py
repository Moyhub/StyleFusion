import argparse
import os
import shutil
import time
import errno
import math

import torch
from torch._C import AggregationType
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import transforms
import torchvision.datasets as datasets
from autoaugment import CIFAR10Policy
from cutout import Cutout
from ISDA import EstimatorCV, ISDALoss

import networks.resnet
import networks.wideresnet
import networks.se_resnet
import networks.se_wideresnet
import networks.densenet_bc
import networks.shake_pyramidnet
import networks.resnext
import networks.shake_shake
import numpy as np
import os
import random
from torch.utils.tensorboard import SummaryWriter
from LTdataset import CIFAR10_truncated

def seed_torch(seed):
    print("此时seed的值为:",seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(description='Using the Style Transfer on Image Classification Task')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')

parser.add_argument('--model', default='resnet', type=str, # resnet
                    help='deep networks to be trained')

parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')

parser.add_argument('--layers', default=110, type=int,
                    help='total number of layers (have to be explicitly given!)')

parser.add_argument('--droprate', default=0.0, type=float,
                    help='dropout probability (default: 0.0)')

parser.add_argument('--augment', type=bool, default=True, help="Whether to use stardard augmentation (default True)")

parser.set_defaults(augment=True) # 设置参数默认值

parser.add_argument('--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--name', default='', type=str,
                    help='name of experiment')
parser.add_argument('--no', default='1', type=str,
                    help='index of the experiment (for recording convenience)')

parser.add_argument('--lambda_0', default=0.5, type=float,
                    help='hyper-patameter_\lambda for ISDA')

# Wide-ResNet & Shake Shake
parser.add_argument('--widen-factor', default=10, type=int,
                    help='widen factor for wideresnet (default: 10)')

# ResNeXt
parser.add_argument('--cardinality', default=8, type=int,
                    help='cardinality for resnext (default: 8)')

# DenseNet
parser.add_argument('--growth-rate', default=12, type=int,
                    help='growth rate for densenet_bc (default: 12)')
parser.add_argument('--compression-rate', default=0.5, type=float,
                    help='compression rate for densenet_bc (default: 0.5)')
parser.add_argument('--bn-size', default=256, type=int,
                    help='cmultiplicative factor of bottle neck layers for densenet_bc (default: 4)')

# Shake_PyramidNet
parser.add_argument('--alpha', default=200, type=int,
                    help='hyper-parameter alpha for shake_pyramidnet')

# Autoaugment
parser.add_argument('--autoaugment', dest='autoaugment', action='store_true',
                    help='whether to use autoaugment')
parser.set_defaults(autoaugment=False)

# cutout
parser.add_argument('--cutout', dest='cutout', action='store_true',
                    help='whether to use cutout')
parser.set_defaults(cutout=False)
parser.add_argument('--n_holes', type=int, default=1,
                    help='number of holes to cut out from image')
parser.add_argument('--length', type=int, default=16,
                    help='length of the holes')

# Cosine learning rate
parser.add_argument('--cos_lr', dest='cos_lr', action='store_true',
                    help='whether to use cosine learning rate')
parser.set_defaults(cos_lr=False)

# Style Transfer
parser.add_argument('-add_style',type=int,default=0,help="with / without the style transfer")
parser.add_argument('-style_type',type=list,default=["Inclass","Outclass"],help="with the Inclass or Outclass to style transfer")
parser.add_argument("-styleratio1",type=float,default=0.2,help="add the mixstyle")
parser.add_argument("-styleratio2",type=float,default=0.2,help="add the mixstyle")
parser.add_argument("-seed",type=int,default=169,help="Set the random seed to retrieval results")
parser.add_argument("-style_position",type=int,default=3,help="where to add the style")

# Setting GPU
parser.add_argument("-GPU",type=int,default="0",help="Setting Gpu")

args = parser.parse_args()
seed_torch(args.seed)

if args.add_style == 1:
    print("Add style")
    args.add_style = True
else:
    args.add_style = False
if args.add_style:
    poslist = [[0],[1],[2],[0,1],[0,2],[1,2],[0,1,2]]
    args.style_position = poslist[args.style_position]
else:
    args.style_position = [0]
# set the GPU
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)
# Configurations adopted for training deep networks.
# (specialized for each type of models)
training_configurations = {
    'resnet': {
        'epochs': 160,
        'batch_size': 128,
        'initial_learning_rate': 0.1,
        'changing_lr': [80, 120],
        'lr_decay_rate': 0.1,
        'momentum': 0.9,
        'nesterov': True,
        'weight_decay': 1e-4,
    },
    'wideresnet': {
        'epochs': 240,
        'batch_size': 128,
        'initial_learning_rate': 0.1,
        'changing_lr': [60, 120, 160, 200],
        'lr_decay_rate': 0.2,
        'momentum': 0.9,
        'nesterov': True,
        'weight_decay': 5e-4,
    },
    'se_resnet': {
        'epochs': 200,
        'batch_size': 128,
        'initial_learning_rate': 0.1,
        'changing_lr': [80, 120, 160],
        'lr_decay_rate': 0.1,
        'momentum': 0.9,
        'nesterov': True,
        'weight_decay': 1e-4,
    },
    'se_wideresnet': {
        'epochs': 240,
        'batch_size': 128,
        'initial_learning_rate': 0.1,
        'changing_lr': [60, 120, 160, 200],
        'lr_decay_rate': 0.2,
        'momentum': 0.9,
        'nesterov': True,
        'weight_decay': 5e-4,
    },
    'densenet_bc': {
        'epochs': 300,
        'batch_size': 64,
        'initial_learning_rate': 0.1,
        'changing_lr': [150, 200, 250],
        'lr_decay_rate': 0.1,
        'momentum': 0.9,
        'nesterov': True,
        'weight_decay': 1e-4,
    },
    'shake_pyramidnet': {
        'epochs': 1800,
        'batch_size': 128,
        'initial_learning_rate': 0.1,
        'changing_lr': [],
        'lr_decay_rate': 0.1,
        'momentum': 0.9,
        'nesterov': True,
        'weight_decay': 1e-4,
    },
    'resnext': {
        'epochs': 350,
        'batch_size': 128,
        'initial_learning_rate': 0.05,
        'changing_lr': [150, 225, 300],
        'lr_decay_rate': 0.1,
        'momentum': 0.9,
        'nesterov': True,
        'weight_decay': 5e-4,
    },
    'shake_shake': {
        'epochs': 1800,
        'batch_size': 64,
        'initial_learning_rate': 0.1,
        'changing_lr': [],
        'lr_decay_rate': 0.1,
        'momentum': 0.9,
        'nesterov': True,
        'weight_decay': 1e-4,
    },
    'shake_shake_x': {
        'epochs': 1800,
        'batch_size': 64,
        'initial_learning_rate': 0.1,
        'changing_lr': [],
        'lr_decay_rate': 0.1,
        'momentum': 0.9,
        'nesterov': True,
        'weight_decay': 1e-4,
    },
}

record_path = './ISDA_test/' + str(args.dataset) \
              + '_' + str(args.model) \
              + '-' + str(args.layers) \
              + (('-' + str(args.widen_factor)) if 'wide' in args.model else '') \
              + (('-' + str(args.widen_factor)) if 'shake_shake' in args.model else '') \
              + (('-' + str(args.growth_rate)) if 'dense' in args.model else '') \
              + (('-' + str(args.alpha)) if 'pyramidnet' in args.model else '') \
              + (('-' + str(args.cardinality)) if 'resnext' in args.model else '') \
              + '_' + str(args.name) \
              + '/' + 'no_' + str(args.no) \
              + '_lambda_0_' + str(args.lambda_0) \
              + ('_standard-Aug_' if args.augment else '') \
              + ('_dropout_' if args.droprate > 0 else '') \
              + ('_autoaugment_' if args.autoaugment else '') \
              + ('_cutout_' if args.cutout else '') \
              + ('_cos-lr_' if args.cos_lr else '') \
              + ('_addstyle_' if args.add_style else "nostyle") \
              + ('_styletype='+"InOutclass"+'_' if args.add_style else '') \
              + (('_styleratio1='+str(args.styleratio1)+"_styleratio2="+str(args.styleratio2)+ "_") if args.add_style else '') \
              + (('_seed='+str(args.seed)+"_")) \
              + (('_layer='+str(args.style_position)) if args.add_style else '')

# 使用Tensorboard
writer = SummaryWriter(log_dir=os.path.join('runs',record_path[12:]))

record_file = record_path + '/training_process.txt'
accuracy_file = record_path + '/accuracy_epoch.txt'
loss_file = record_path + '/loss_epoch.txt'
check_point = os.path.join(record_path, args.checkpoint)

def main():

    global best_prec1
    best_prec1 = 0

    global val_acc
    val_acc = []

    global class_num

    class_num = args.dataset == 'cifar10' and 10 or 100

    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    if args.augment:
        if args.autoaugment:
            print('Autoaugment')
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                                  (4, 4, 4, 4), mode='reflect').squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(), CIFAR10Policy(),
                transforms.ToTensor(),
                Cutout(n_holes=args.n_holes, length=args.length),
                normalize,
            ])

        elif args.cutout:
            print('Cutout')
            transform_train = transforms.Compose([
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

        else:
            print('Standrad Augmentation!')
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                                  (4, 4, 4, 4), mode='reflect').squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
    else:
        transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

    kwargs = {'num_workers': 1, 'pin_memory': True}
    assert(args.dataset == 'cifar10' or args.dataset == 'cifar100')
    dataidxs = []
    with open("./data/dataidx.txt", "r") as f:
        for line in f:
            dataidxs.append(int(line.strip()))
    train_ds = CIFAR10_truncated("./data/cifar10", dataidxs=dataidxs, train=True, transform=transform_train, download=True)
    test_ds = CIFAR10_truncated("./data/cifar10", train=False, transform=transform_test, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_ds,batch_size=training_configurations[args.model]['batch_size'],shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=test_ds,batch_size=training_configurations[args.model]['batch_size'],shuffle=True)

    # create model
    # eval() 函数用来执行一个字符串表达式，并返回表达式的值。
    if args.model == 'resnet':
        model = eval('networks.resnet.resnet' + str(args.layers) + '_cifar')(num_classes=class_num,add_style=args.add_style,styleratio1=args.styleratio1,styleratio2=args.styleratio2,style_type=args.style_type,style_position=args.style_position,dropout_rate=args.droprate,seed=args.seed) 
    elif args.model == 'se_resnet':
        model = eval('networks.se_resnet.resnet' + str(args.layers) + '_cifar')(num_classes=class_num,add_style=args.add_style,styleratio1=args.styleratio1,styleratio2=args.styleratio2,style_type=args.style_type,style_position=args.style_position,dropout_rate=args.droprate,seed=args.seed)
    elif args.model == 'wideresnet':
        model = networks.wideresnet.WideResNet(args.layers, args.dataset == 'cifar10' and 10 or 100,
                            args.widen_factor, dropRate=args.droprate,add_style=args.add_style,styleratio1=args.styleratio1,styleratio2=args.styleratio2,style_type=args.style_type,style_position=args.style_position,seed=args.seed)
    elif args.model == 'se_wideresnet':
        model = networks.se_wideresnet.WideResNet(args.layers, args.dataset == 'cifar10' and 10 or 100,
                            args.widen_factor, dropRate=args.droprate)
    elif args.model == 'densenet_bc':
        # CIFAR100时候需要改动内部的num_class
        model = networks.densenet_bc.DenseNet(numclass=class_num,
                                              growth_rate=args.growth_rate,
                                              block_config=(int((args.layers - 4) / 6),) * 3,
                                              compression=args.compression_rate,
                                              num_init_features=24,
                                              bn_size=args.bn_size,
                                              drop_rate=args.droprate,
                                              small_inputs=True,
                                              efficient=False,add_style=args.add_style,styleratio1=args.styleratio1,styleratio2=args.styleratio2,style_type=args.style_type,style_position=args.style_position,seed=args.seed)
    elif args.model == 'shake_pyramidnet':
        model = networks.shake_pyramidnet.PyramidNet(dataset=args.dataset, depth=args.layers, alpha=args.alpha, num_classes=class_num, bottleneck = True)
    elif args.model == 'resnext':
        if args.cardinality == 8:
            model = networks.resnext.resnext29_8_64(class_num,add_style=args.add_style,styleratio1=args.styleratio1,styleratio2=args.styleratio2,style_type=args.style_type,style_position=args.style_position,dropout_rate=args.droprate,seed=args.seed)
        if args.cardinality == 16:
            model = networks.resnext.resnext29_16_64(class_num,add_style=args.add_style,styleratio1=args.styleratio1,styleratio2=args.styleratio2,style_type=args.style_type,style_position=args.style_position,dropout_rate=args.droprate,seed=args.seed)
    elif args.model == 'shake_shake':
        if args.widen_factor == 112:
            model = networks.shake_shake.shake_resnet26_2x112d(class_num)
        if args.widen_factor == 32:
            model = networks.shake_shake.shake_resnet26_2x32d(num_classes=class_num,add_style=args.add_style,styleratio1=args.styleratio1,styleratio2=args.styleratio2,style_type=args.style_type,style_position=args.style_position,seed=args.seed)
        if args.widen_factor == 96:
            model = networks.shake_shake.shake_resnet26_2x64d(class_num)
    elif args.model == 'shake_shake_x':

        model = networks.shake_shake.shake_resnext29_2x4x64d(class_num)

    if not os.path.isdir(check_point):
        mkdir_p(check_point)

    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])
    ))

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    ce_criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=training_configurations[args.model]['initial_learning_rate'],
                                momentum=training_configurations[args.model]['momentum'],
                                nesterov=training_configurations[args.model]['nesterov'],
                                weight_decay=training_configurations[args.model]['weight_decay'])

    model = torch.nn.DataParallel(model).cuda()

    if args.resume: # 默认是不中断后重新开始
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        val_acc = checkpoint['val_acc']
        best_prec1 = checkpoint['best_acc']
        np.savetxt(accuracy_file, np.array(val_acc))
    else:
        start_epoch = 0

    for epoch in range(start_epoch, training_configurations[args.model]['epochs']):

        adjust_learning_rate(optimizer, epoch + 1)

        # train for one epoch
        train(train_loader, model, ce_criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, ce_criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_prec1,
            'optimizer': optimizer.state_dict(),
            'val_acc': val_acc,

        }, is_best, checkpoint=check_point)
        print('Best accuracy: ', best_prec1)
        # print("Test_Accuracy",prec1)
        np.savetxt(accuracy_file, np.array(val_acc))  
        print(100 * "*")

    writer.close()
    print('Best accuracy: ', best_prec1)
    print('Average accuracy', sum(val_acc[len(val_acc) - 10:]) / 10)
    np.savetxt(accuracy_file, np.array(val_acc))

def train(train_loader, model, criterion, optimizer, epoch):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    train_batches_num = len(train_loader)

    ratio = args.lambda_0 * (epoch / (training_configurations[args.model]['epochs']))
    # switch to train mode
    model.train()

    end = time.time()
    for i, (x, target) in enumerate(train_loader):
        target = target.cuda()
        x = x.cuda()
        input_var = torch.autograd.Variable(x)
        target_var = torch.autograd.Variable(target)

        # 此处加入针对CIFAR10的损失计算
        output = model(input_var,target_var)
        loss = criterion(output,target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), x.size(0))
        top1.update(prec1.item(), x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()


        if (i+1) % args.print_freq == 0:
            # print(discriminate_weights)
            fd = open(record_file, 'a+')
            string = ('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                      'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
                      'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(
                       epoch, i+1, train_batches_num, batch_time=batch_time,
                       loss=losses, top1=top1))

            print(string)
            fd.write(string + '\n')
            fd.close()

def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    train_batches_num = len(val_loader)

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        with torch.no_grad():
            output = model(input_var)

        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0:
            fd = open(record_file, 'a+')
            string = ('Test: [{0}][{1}/{2}]\t'
                      'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                      'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
                      'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(
                       epoch, (i+1), train_batches_num, batch_time=batch_time,
                       loss=losses, top1=top1))
            print(string)
            fd.write(string + '\n')
            fd.close()

    fd = open(record_file, 'a+')
    string = ('Test: [{0}][{1}/{2}]\t'
              'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
              'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
              'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(
        epoch, (i + 1), train_batches_num, batch_time=batch_time,
        loss=losses, top1=top1))
    print(string)
    fd.write(string + '\n')
    fd.close()
    val_acc.append(top1.ave)
    writer.add_scalar("Test_Accuracy",top1.ave,epoch)

    return top1.ave

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.ave = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.ave = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate"""
    if not args.cos_lr:
        if epoch in training_configurations[args.model]['changing_lr']:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= training_configurations[args.model]['lr_decay_rate']

    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.5 * training_configurations[args.model]['initial_learning_rate']\
                                * (1 + math.cos(math.pi * epoch / training_configurations[args.model]['epochs']))


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
