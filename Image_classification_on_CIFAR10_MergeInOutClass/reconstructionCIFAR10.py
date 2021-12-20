#! /bin/env/python3

import torch 
import torch.nn.functional as F
from torch.nn.functional import interpolate
from torchvision import datasets, transforms
import matplotlib
import torch.nn as nn
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# In this configuration of encoder we do not feed matrices of shape (subj, ims, features), 
# but raw eeg of shape (subj, ims, ch, time)
# This is not directly comparable to previous analyis where DNN got input of shape (subj, ims, features)
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
class conv_autoencoder_raw(torch.nn.Module):
    '''
    NB! DNN works on raw EEG (subj, im, CH, TIME) insteas of (subj, im ,feat).
    Input shall be of shape (batch_size, subj, eeg_ch, eeg_time)
    Attributes:
        n_subj -int, n sunjects
        out_ch1, 2 ,3 - int, number of output channels for each conv layer of encoder
        feat_dims1, 2 ,3 - tuple of ints, feature dimensions at each conv layer
        p - float, dropout_probability. Default 0 (no dropout)
        spefified dimensions. If True, does not resample. Default=True.
        enc_layer -int, which encoder layer to resurn in the output. Default=2 (last layer)
    Methods:
        forward. 
            Outputs:
            enc (ims, 1, feature1, feature2)
            dec (ims, subj, eeg_ch, eeg_time)
    '''
    def __init__(self, inp_chs=3, enc_ch1=16, enc_ch2=32, enc_ch3=64,\
                dec_ch1 = 64, dec_ch2 = 32, \
                enc_dims1=(8,12), enc_dims2 = (4,4), enc_dims3=(8,12), \
                out_dims = None,\
                p=0):
         
        conv1 = torch.nn.Sequential(torch.nn.Conv2d(inp_chs, enc_ch1, 3), \
                                torch.nn.BatchNorm2d(enc_ch1),\
                                torch.nn.ReLU(),\
                                torch.nn.Dropout2d(p=p))
        conv2 = torch.nn.Sequential(torch.nn.Conv2d(enc_ch1, enc_ch2, 3), \
                                torch.nn.BatchNorm2d(enc_ch2),\
                                torch.nn.ReLU(),\
                                torch.nn.Dropout2d(p=p))
        conv3 = torch.nn.Sequential(torch.nn.Conv2d(enc_ch2, enc_ch3, 3), \
                                torch.nn.BatchNorm2d(enc_ch3),\
                                torch.nn.Sigmoid())
        deconv1 = torch.nn.Sequential(torch.nn.ConvTranspose2d(enc_ch3, dec_ch1, 3), \
                                torch.nn.BatchNorm2d(dec_ch1),\
                                torch.nn.ReLU())
        deconv2 = torch.nn.Sequential(torch.nn.ConvTranspose2d(dec_ch1, dec_ch2, 3), \
                                torch.nn.BatchNorm2d(dec_ch2),\
                                torch.nn.ReLU())
        deconv3 = torch.nn.Sequential(torch.nn.ConvTranspose2d(dec_ch2, inp_chs, 3), \
                                torch.nn.BatchNorm2d(inp_chs),\
                                torch.nn.Sigmoid())
        super(conv_autoencoder_raw, self).__init__()

        self.conv1 = conv1
        self.conv2 = conv2
        self.conv3 = conv3

        self.deconv1 = deconv1
        self.deconv2 = deconv2
        self.deconv3 = deconv3
        
        self.encoder = torch.nn.Sequential(conv1, conv2, conv3)
        self.decoder = torch.nn.Sequential(deconv1, deconv2, deconv3)

    def forward(self, data):
        # encoder
        #orig_dims = data.shape[-2:]
        #batch_size = data.shape[0]
        #out1 = self.conv1(data)
        #    out1 = F.interpolate(out1, self.feat_dims1)
        #out2 = self.conv2(out1)
        #if not self.no_resampling:
        #    out2 = F.interpolate(out2, self.feat_dims2)
        #out3 = self.conv3(out2)
        #if not self.no_resampling:
        #    out3 = F.interpolate(out3, self.feat_dims3)
        # decoder
        #enc_layers=[out1, out2, out3]
        #enc_out=enc_layers[self.enc_layer]
        #dec_out = self.decoder(out3)
        #dec_out = F.interpolate(dec_out, orig_dims)

        orig_dims = data.shape[-2:]
        enc_out = self.encoder(data)
        dec_out = self.decoder(enc_out)
        if not dec_out[-2:] == orig_dims:
            dec_out = F.interpolate(dec_out, orig_dims)
        return enc_out, dec_out

def imshow(im):
    im = im/2 +0.5
    plt.imshow(np.transpose(im, (1,2,0)))


if __name__=='__main__':
    import argparse
    import time
    from collections import defaultdict
    import matplotlib.pyplot as plt
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument('-n_workers', type=int, default=0, help='default=0')
    parser.add_argument('-batch_size', type=int, default=16, help='default=16')
    parser.add_argument('-gpu', default=True, help='Flag, whether to '
    'use GPU. Default = False.')
    parser.add_argument('-lr', type=float, default=0.001, help='Learning rate. Default=0.001')
    parser.add_argument('-n_epochs',type=int, default=10, help='How many times to pass '
    'through the dataset. Default=10')
    parser.add_argument('-bpl', type=int, default=20, help='Save loss every '
    'bacth_per_loss mini-batches. Default=20.')
    parser.add_argument('-epta', type=int, default=1, help='Save test '
    'set accuracy every epochs_per_test_accuracy  epochs. Default == 1')
    parser.add_argument('-enc_chs', nargs=5, type=int, default = [16, 32, 64, 128, 256], help=\
    'output channels for encoder blocks 1-5. Default = [16, 32, 64, 128, 256].')
    parser.add_argument('-enc_out', type=int, default= 512, help='encoder final layer '
    'output channels. Default=512.')
    parser.add_argument('-dec_chs', nargs=4, type=int, default=[128, 64, 32, 16], help=\
    'output channels for decoder blocks 1-4. Default = [128, 64, 32, 16].')
    parser.add_argument('-p_enc', type=float, default=0., help='Dropout value for encoder blocks 1-4. '
    'Default=0.')
    args = parser.parse_args()


    # create datasets
    # EEG datasets
    
    transform = transforms.ToTensor()
    
    dataset_train =  datasets.CIFAR10(root='data',\
                              train=True, download=True, transform=transform)
    dataset_test = datasets.CIFAR10(root='data',\
                             train=False, download=True, transform=transform)

    train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,\
                                                    shuffle=True, num_workers=args.n_workers,\
                                                    drop_last=False)

    test_dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,\
                                                    shuffle=False, num_workers=args.n_workers,\
                                                    drop_last=False)

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

  
    # define the model
    ch_time = (32, 32)
    model = conv_autoencoder_raw()
        
    if args.gpu and args.n_workers >=1:
        warnings.warn('Using GPU and n_workers>=1 can cause some difficulties.')
    if args.gpu:
        device_name="cuda"
        device=torch.device("cuda:0")
        model.to(device)
        # model = model.cuda()
        model=torch.nn.DataParallel(model)
        print("Using "+str(torch.cuda.device_count()) +" GPUs.")
    elif not args.gpu:
        device=torch.device("cpu")
        device_name="cpu"
        print("Using CPU.")

    # define the loss
    loss_fn = torch.nn.MSELoss() 

    # define optmizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

    # Loss and  accuracy init 
    losses = defaultdict()
    cntr_epta=0 

    eval = True
    if eval:
        model = torch.load("encdec.pkl")

        dataiter = iter(test_dataloader)
        images, labels = dataiter.next()
        images = images.cuda()
        enc, dec = model(images)

        content_out = model.module.conv1(images)

        style_image, style_label = dataiter.next()
        style_image = style_image.cuda()
        styleout = model.module.conv1(style_image)

        norm2d = Adain2d(alpha1=1.0,alpha2=0)
        # 第一层调制结果
        out = norm2d(content_out,styleout,styleout)

        content_out = model.module.conv2(out)
        styleout = model.module.conv2(styleout)
        # 第二层调制结果
        out = norm2d(content_out,styleout,styleout)

        out = model.module.conv3(out)
        out = model.module.decoder(out)

        out = out.cpu().detach().numpy()
        images = images.cpu().detach().numpy()
        style_image = style_image.cpu().detach().numpy()

        print("Original Images")
        fig, axes = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(12,4))
        for idx in np.arange(5):
            ax = fig.add_subplot(1, 5, idx+1, xticks=[], yticks=[])
            imshow(images[idx])
            ax.set_title(classes[labels[idx]])
        plt.savefig("OriginImage.jpg")

        print("Style Images")
        fig, axes = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(12,4))
        for idx in np.arange(5):
            ax = fig.add_subplot(1, 5, idx+1, xticks=[], yticks=[])
            imshow(style_image[idx])
            ax.set_title(classes[style_label[idx]])
        plt.savefig("StyleImage.jpg")
        
        print('Reconstructed Images')
        fig, axes = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(12,4))
        for idx in np.arange(5):
            ax = fig.add_subplot(1, 5, idx+1, xticks=[], yticks=[])
            imshow(out[idx])
            ax.set_title(classes[labels[idx]])
        plt.savefig("Reconstruction.jpg")



    else:
        # Loop through EEG dataset in batches
        for epoch in range(args.n_epochs):
            model.train()
            cntr=0
            tic = time.time()
            losses["epoch"+str(epoch)]=[]

            for images, labels in train_dataloader:
                if args.gpu:
                    images = images.to(device)
                enc, dec = model.forward(images)
            
                # import ipdb; ipdb.set_trace()
                # compute loss - minimize diff between outputs of net and real data?
                loss = loss_fn(dec, images)
                losses["epoch"+str(epoch)].append(loss.cpu().detach().numpy())

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                # save loss every bpl mini-batches
                if cntr % args.bpl == 0 and cntr != 0:
                    print('Epoch {:d}, average loss over bacthes {:d} - {:d}: {:.3f}.'.\
                    format(epoch, int(cntr-args.bpl), cntr, sum(losses["epoch"+str(epoch)][-args.bpl:])/args.bpl))
                cntr+=1

            # save test accuracy every epta epcohs
            if cntr_epta % args.epta == 0:
                
                dataiter = iter(test_dataloader)
                images, labels = dataiter.next()
                enc, dec = model(images)
                dec = dec.cpu().detach().numpy()
                
                #Original Images
                print("Original Images")
                fig, axes = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(12,4))
                for idx in np.arange(5):
                    ax = fig.add_subplot(1, 5, idx+1, xticks=[], yticks=[])
                    imshow(images[idx])
                    ax.set_title(classes[labels[idx]])
                if epoch == args.n_epochs-1:
                    plt.savefig("OriginImage"+str(cntr_epta)+".jpg")
                    plt.close()
                
                #Reconstructed Images
                print('Reconstructed Images')
                fig, axes = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(12,4))
                for idx in np.arange(5):
                    ax = fig.add_subplot(1, 5, idx+1, xticks=[], yticks=[])
                    imshow(dec[idx])
                    ax.set_title(classes[labels[idx]])
                # plt.show() 
                if epoch == args.n_epochs-1:
                    plt.savefig("ReconstructedImage"+str(cntr_epta)+".jpg")
                    plt.close()
                # inp = input('Next?')
                # if inp == 'y':
                #     pass
                # elif inp == 'n':
                #     break

                cntr_epta += 1
            
            if epoch == args.n_epochs-1:
                torch.save(model,"encdec.pkl")

