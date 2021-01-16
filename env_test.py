#envi_test.py
#used to examine results more closely

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import numpy as np

from mpl_toolkits.mplot3d import Axes3D  # for 3d projection plots

# ======= CONSTANTS ==========
EPOCHS = 500000
FULL_BATCH = True
BATCH_SIZE = 64 #overridden if FULL_BATCH
LEARNING_RATE = 0.00003
TRAIN_RATIO = 0.75
ENV_SIZE = 4

desc_str = "linear_model_TR_"+str(int(TRAIN_RATIO*100))

# ======== NETWORK DEFINITION ====
layer_str = "lr_5e-05_ep_20000_bs_64_L_3_16_32_92_state_dict"
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(3,16),
            nn.ReLU(),
            nn.Linear(16,32),
            nn.ReLU(),
            nn.Linear(32,92)
        )
    
    def forward(self,x):
        results = self.net(x)
        return results

class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(3,16),
            nn.ReLU(),
            nn.Linear(16,32),
            nn.ReLU(),
            nn.Linear(32,92-20)
        )
    
    def forward(self,x):
        results = self.net(x)
        return results

# ======= LOAD DATA ==========
in_path = "combined_data.csv"
raw_data = np.genfromtxt(in_path,delimiter=',')
times = raw_data[1:,0]
N_env = len(times)
env_data = np.zeros((N_env,ENV_SIZE))
env_data[:,0] = np.log10(raw_data[1:,1]) #log10 illuminance
#env_data[:,0] = raw_data[1:,1] #illuminance
env_data[:,1] = raw_data[1:,3] #temperature
env_data[:,2] = raw_data[1:,4] #humidity
env_data[:,3] = 1
wfs = raw_data[1:,5:]
wfs_p = wfs[:,20:]

net_env_data = np.zeros((N_env,3))
net_env_data[:,0] = raw_data[1:,1] #illuminance
net_env_data[:,1] = raw_data[1:,3] #temperature
net_env_data[:,2] = raw_data[1:,4] #humidity

#============ SPLIT DATA ==============
x_full = env_data
border = int(len(x_full)*TRAIN_RATIO)
x_train = x_full[0:border]
x_test = x_full[border:]

y_full = wfs
y_train = y_full[0:border]
y_test = y_full[border:]

N_train = len(x_train)
N_test = len(x_test)
N_full = N_train + N_test
assert(N_full == len(x_full))
train_indices = list(range(N_train))


#========== LINEAR MODEL =============
#linear model plot
#Nx92 = Nx4 @ 4x92
#Y = XM
#X+Y = M
M = np.linalg.pinv(x_train) @ y_train
lp_full = x_full @ M
l_cc = np.zeros(N_test+N_train)
for i in range(N_full):
    l_cc[i] = np.corrcoef(y_full[i], lp_full[i])[0,1]
"""
plt.figure()
plt.plot(l_cc[0:N_train], label="Training")
plt.plot(range(N_train, N_full), l_cc[N_train:N_full], label="Testing")
plt.plot(env_data[:,0]/max(env_data[:,0]), alpha=0.5, label="Illuminance")
plt.title("Linear Model Prediction CC")
plt.ylim((0,1))
plt.ylabel("CC")
plt.xlabel("Sample")
plt.legend()
plt.savefig("plots/"+desc_str+"_cc_full")


#zoomed
plt.figure()
plt.plot(l_cc[0:N_train], label="Training")
plt.plot(range(N_train, N_full), l_cc[N_train:N_full], label="Testing")
ylims = plt.ylim()
norm_illuminance = np.array(env_data[:,0])
norm_illuminance = norm_illuminance/max(norm_illuminance)*(ylims[1]-ylims[0])+ylims[0]
plt.plot(norm_illuminance,label="Illuminance",alpha=0.5)
plt.title("Zoomed Linear Model Prediction CC")
plt.ylabel("CC")
plt.xlabel("Sample")
plt.legend()
plt.savefig("plots/"+desc_str+"_cc_full_zoomed")
"""
"""
#3d scatter plot of data
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(x_train[:,0],x_train[:,1],x_train[:,2],label="Train",alpha=0.5)
ax.scatter(x_test[:,0],x_test[:,1],x_test[:,2],label="Test",alpha=0.5)
plt.title("Environmental Data Split")
ax.set_xlabel("log10 Illuminance (log10 Lux)")
ax.set_ylabel("Temperature (F)")
ax.set_zlabel("Relative Humidity (%)")
plt.legend()
"""

#best un-pruned NN results: lr_5e-05_ep_20000_bs_64_L_3_16_32_92

#load best network
network = Net().double()
network.load_state_dict(torch.load("models/lr_5e-05_ep_20000_bs_64_L_3_16_32_92_state_dict"))
network.double()
network.eval()
x_full_tensor = torch.from_numpy(net_env_data).view(-1,3).double()
net_predictions_full = network(x_full_tensor).detach().numpy()
#get CC's
net_cc = np.zeros(N_full)
for i in range(N_full):
    net_cc[i] = np.corrcoef(y_full[i], net_predictions_full[i])[0,1]
#get CC's for pruned area
net_cc_p = np.zeros(N_full)
for i in range(N_full):
    net_cc_p[i] = np.corrcoef(y_full[i,20:], net_predictions_full[i,20:])[0,1]


#load best pruned network
network_p = PNet().double()
network_p.load_state_dict(torch.load("models/lr_5e-05_ep_20000_bs_64_L_3_16_32_72_log_F_norm_F_prune_T_state_dict"))
network_p.double()
network_p.eval()
net_p_predictions = network_p(x_full_tensor).detach().numpy()
#get CC's for pruned area
net_p_cc_p = np.zeros(N_full)
for i in range(N_full):
    net_p_cc_p[i] = np.corrcoef(wfs_p[i], net_p_predictions[i])[0,1]
#get "full CC"
wf_prefix = np.mean(wfs[:,0:20],axis=0)
net_p_cc = np.zeros(N_full)
for i in range(N_full):
    net_p_cc[i] = np.corrcoef(wfs[i], np.concatenate((wf_prefix, net_p_predictions[i])))[0,1]


plt.figure()
ax = plt.gca()
plt.title("Full Waveform R.C.C. Comparison")
plt.plot(l_cc, label="Linear Model, Unpruned Solution")#, lw=1)
plt.plot(net_cc, label="N.N., Unpruned During Training")#, lw=1)
plt.plot(net_p_cc, label = "N.N., Pruned During Training")#, lw=1)

#ylims = plt.ylim((0.8,1))
ylims = plt.ylim()
norm_illuminance = np.array(env_data[:,0])
norm_illuminance -= min(norm_illuminance)
norm_illuminance = 0.5*norm_illuminance/max(norm_illuminance)*(ylims[1]-ylims[0])+ylims[0]
plt.plot(norm_illuminance,":",label="Normalized Log10 Illuminance",alpha=0.5)
plt.xlabel("Data Point")
plt.ylabel("Reconstruction Correlation Coefficient")
handles, labels = ax.get_legend_handles_labels()
handles.reverse()
labels.reverse()
handles = list(np.roll(handles,-1))
labels= list(np.roll(labels,-1))
plt.legend(handles,labels, loc='lower left')
plt.savefig("plots/full_waveform_prune_comparison")
"""
plt.figure()
plt.title("Pruned Region C.C.")
plt.plot(net_cc_p, label="Unpruned During Training")
plt.plot(net_p_cc_p, label = "Pruned During Training")
ylims = plt.ylim((0.8,1))
norm_illuminance = np.array(env_data[:,0])
norm_illuminance -= min(norm_illuminance)
norm_illuminance = norm_illuminance/max(norm_illuminance)*(ylims[1]-ylims[0])+ylims[0]
plt.plot(norm_illuminance,":",label="Illuminance",alpha=0.5)
plt.legend(loc='lower left')
plt.savefig("plots/pruned_region_prune_comparison")
"""

plt.figure()
plt.hist(l_cc, 100, label="Linear Model, Unpruned Solution", alpha = 0.7)
plt.hist(net_cc, 100, label="N.N., Unpruned During Training", alpha = 0.7)
plt.hist(net_p_cc, 100, label = "N.N., Pruned During Training", alpha = 0.7)
plt.legend(loc="upper left")
plt.title("R.C.C. Histograms")
plt.xlabel("Reconstruction Correlation Coefficient")
plt.ylabel("Count")
plt.text(0.05,0.3,"Pruned Network:\n   Mean: {0:.4f}\n   Std. Dev.: {1:.5f}".format(np.mean(net_p_cc),np.std(net_p_cc)),transform=ax.transAxes,fontsize=10)
plt.text(0.05,0.46,"Unpruned Network:\n   Mean: {0:.4f}\n   Std. Dev.: {1:.5f}".format(np.mean(net_cc),np.std(net_p_cc)),transform=ax.transAxes,fontsize=10)
plt.text(0.05,0.62,"Linear Model:\n   Mean: {0:.4f}\n   Std. Dev.: {1:.5f}".format(np.mean(l_cc),np.std(l_cc)),transform=ax.transAxes,fontsize=10)
plt.savefig("plots/full_waveform_prune_comparison_hists")

plt.show()