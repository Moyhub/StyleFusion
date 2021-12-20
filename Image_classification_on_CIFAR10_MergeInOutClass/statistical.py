import pandas as pd
import numpy as np
import os 

dir = "ISDA_test/cifar10_resnet-32_"
# dir = "ISDA_test/cifar10_resnet-110_"
# dir = "ISDA_test/cifar10_se_resnet-110_"
# dir = "ISDA_test/cifar10_wideresnet-16-8_"
# dir = "ISDA_test/cifar100_resnet-32_"
# dir = "ISDA_test/cifar100_se_resnet-110_"
# seed=(169 196 256 289 324 400 529 676 841 1024)
# col_index = [0.2,0.5,0.8,1.0]
col_index = [0.1,0.2,0.3,0.4]
raw_index = [0.1,0.2,0.3,0.4]

file1 = np.zeros([4,4])
file2 = np.zeros([4,4])
file3 = np.zeros([4,4])
file4 = np.zeros([4,4])
file5 = np.zeros([4,4])
file6 = np.zeros([4,4])
file7 = np.zeros([4,4])
file8 = np.zeros([4,4])
file9 = np.zeros([4,4])
file10 = np.zeros([4,4])

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
        if "addstyle" in dirs:
            col = col_index.index(float(dirs[dirs.find("styleratio1")+12:dirs.find("styleratio1")+15])) # 列号
            raw = raw_index.index(float(dirs[dirs.find("styleratio2")+12:dirs.find("styleratio2")+15])) 
            # if "Outclass" in dirs:
            #     raw = raw + 4
            if "seed=169" in dirs:
                file1[raw,col] = finalacc
            elif "seed=196" in dirs:
                file2[raw,col] = finalacc
            elif "seed=256" in dirs:
                file3[raw,col] = finalacc
            elif "seed=289" in dirs:
                file4[raw,col] = finalacc
            elif "seed=324" in dirs:
                file5[raw,col] = finalacc
            elif "seed=400" in dirs:
                file6[raw,col] = finalacc
            elif "seed=529" in dirs:
                file7[raw,col] = finalacc
            elif "seed=676" in dirs:
                file8[raw,col] = finalacc
            elif "seed=841" in dirs:
                file9[raw,col] = finalacc
            elif "seed=1024" in dirs:
                file10[raw,col] = finalacc     
        else:
            origin.append(finalacc)

np.savetxt("result/seed169.csv",file1,fmt="%.2f",delimiter=",")
np.savetxt("result/seed196.csv",file2,fmt="%.2f",delimiter=",")
np.savetxt("result/seed256.csv",file3,fmt="%.2f",delimiter=",")
np.savetxt("result/seed289.csv",file4,fmt="%.2f",delimiter=",")
np.savetxt("result/seed324.csv",file5,fmt="%.2f",delimiter=",")
np.savetxt("result/seed400.csv",file6,fmt="%.2f",delimiter=",")
np.savetxt("result/seed529.csv",file7,fmt="%.2f",delimiter=",")
np.savetxt("result/seed676.csv",file8,fmt="%.2f",delimiter=",")
np.savetxt("result/seed841.csv",file9,fmt="%.2f",delimiter=",")
np.savetxt("result/seed1024.csv",file10,fmt="%.2f",delimiter=",")

np.savetxt("result/origin.csv",np.array(origin),fmt="%.2f",delimiter=",")

