import pandas as pd
import numpy as np
import os 

# dir = "ISDA_test/cifar10_resnet-32_"
dir = "ISDA_test/cifar10_resnet-110_"
# dir = "ISDA_test/cifar10_se_resnet-110_"
# dir = "ISDA_test/cifar10_wideresnet-16-8_"
# dir = "ISDA_test/cifar100_resnet-32_"
# dir = "ISDA_test/cifar100_se_resnet-110_"
# seed=(169 196 256 289 324 400 529 676 841 1024)
# col_index = [0.2,0.5,0.8,1.0]

origin = []
new_dir = os.listdir(dir)
new_dir.sort()

for dirs in new_dir:
    print("当前读取目录：",dirs)
    dirs  = os.path.join(dir,dirs)
    for file in os.listdir(dirs):
        if file == "accuracy_epoch.txt":
            testfile = os.path.join(dirs,file)
            acc = np.loadtxt(testfile)
            finalacc = acc[-1]   
        else:
            continue   
        
        origin.append(finalacc)

np.savetxt("result/origin.csv",np.array(origin),fmt="%.2f",delimiter=",")

