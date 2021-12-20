import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# For Cutout
origin = 100 - np.loadtxt("ISDA_test/cifar10_resnet-110_/no_1_lambda_0_0.5_standard-Aug_nostyle_seed=289_/accuracy_epoch.txt")
origin_fusestyle = 100 - np.loadtxt("ISDA_test/cifar10_resnet-110_/no_1_lambda_0_0.5_standard-Aug__addstyle__styletype=InOutclass__styleratio1=0.4_styleratio2=0.2__seed=289__layer=[0, 1]/accuracy_epoch.txt")
origin_cut = 100 - np.loadtxt("lossresults/AA.txt")
origin_cut_fusestyle = 100 - np.loadtxt("lossresults/AAFuseStyle.txt")

window_siez = 40
origin = origin[window_siez:]
origin_fusestyle = origin_fusestyle[window_siez:]
origin_cut = origin_cut[window_siez:]
origin_cut_fusestyle = origin_cut_fusestyle[window_siez:]

print(origin[-1],origin_fusestyle[-1],origin_cut[-1],origin_cut_fusestyle[-1])

xaxis = np.array(range(len(origin)))+window_siez

plt.figure()
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.plot(xaxis,origin)
plt.plot(xaxis,origin_fusestyle)
plt.plot(xaxis,origin_cut)
plt.plot(xaxis,origin_cut_fusestyle)
plt.grid(linestyle="-.")
plt.xlim(window_siez,160)
plt.ylim(2,20)
plt.xlabel("Epoch")
plt.ylabel("Test Error Rate (%)")
plt.legend(["ResNet110","ResNet110+FuseStyle","ResNet110+AA","ResNet110+AA+FuseStyle"])
plt.savefig("losscutout.pdf")