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

# ======= LOAD DATA ==========
in_path = "combined_data.csv"
raw_data = np.genfromtxt(in_path,delimiter=',')
times = raw_data[1:,0]
N_env = len(times)
env_data = np.zeros((N_env,ENV_SIZE))
env_data[:,0] = np.log10(raw_data[1:,1]) #log10 illuminance                        m
#env_data[:,0] = raw_data[1:,1] #illuminance
env_data[:,1] = raw_data[1:,3] #temperature
env_data[:,2] = raw_data[1:,4] #humidity
env_data[:,3] = 1
wfs = raw_data[1:,5:]

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

#best NN results: lr_5e-05_ep_20000_bs_64_L_3_16_32_92_cc_full


#load best network
network = Net().double()
network.load_state_dict(torch.load("models/lr_5e-05_ep_20000_bs_64_L_3_16_32_92_state_dict"))
network.double()
network.eval()
x_full_tensor = torch.from_numpy(net_env_data).view(-1,3).double()
net_predictions_full = network(x_full_tensor).detach().numpy()
#verify that cc looks the same
net_cc = np.zeros(N_full)
for i in range(N_full):
    net_cc[i] = np.corrcoef(y_full[i], net_predictions_full[i])[0,1]
plt.figure()
plt.plot(net_cc)
plt.title("loaded network CC verification")


#get MSEs
lp_mses = np.mean((y_full - lp_full)**2, axis=1)
net_mses = np.mean((y_full - net_predictions_full)**2, axis=1)


plt.show()