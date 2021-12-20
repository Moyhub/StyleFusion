# import matplotlib.pyplot as plt
import cv2
import torch
# from torch._C import uint8
import torchvision.transforms as transforms
from cutout import Cutout
import torch.nn.functional as F
from PIL import Image
import numpy as np
from autoaugment import CIFAR10Policy,ImageNetPolicy
# import torch

# img = cv2.imread("car3.jpg")
img = Image.open('car3.jpg') 
img = img.resize((180,180))
# print(img.shape)
# img = torch.from_numpy(img)

normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                        (4, 4, 4, 4), mode='reflect').squeeze()),
    transforms.ToPILImage(),
    # transforms.RandomCrop(128),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    Cutout(n_holes=1, length=80),
    # normalize,
])

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                        (4, 4, 4, 4), mode='reflect').squeeze()),
    transforms.ToPILImage(),
    # transforms.RandomCrop(32),
    transforms.RandomHorizontalFlip(), ImageNetPolicy(),# CIFAR10Policy(),
    transforms.ToTensor(),
    # Cutout(n_holes=args.n_holes, length=args.length),
    normalize,
])

img = transform_train(img)

img = img.numpy().transpose(1,2,0)*255
img = np.uint8(img)
img = cv2.resize(img,(300,180))

cv2.imwrite("cat.jpg",img)

# print(img.size)

# Cut

