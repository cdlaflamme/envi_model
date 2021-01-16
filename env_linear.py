#env_linear.py
#linear model to reconstruct sstdr waveforms using environment data

import matplotlib.pyplot as plt
import random
import numpy as np

from mpl_toolkits.mplot3d import Axes3D  # for 3d projection plots

# ======= CONSTANTS ==========

TRAIN_RATIO = 0.75
ENV_SIZE = 4

desc_str = "linear_model_TR_"+str(int(TRAIN_RATIO*100))

# ======= LOAD DATA ==========
in_path = "combined_data_new.csv"
raw_data = np.genfromtxt(in_path,delimiter=',',skip_header=1)
times = raw_data[1:,0]
N_env = len(times)
env_data = np.zeros((N_env,ENV_SIZE))

env_data[:,0] = np.where(raw_data[1:,1]==0, 0.01, raw_data[1:,1]) #special case for 0 lux
env_data[:,0] = np.log10(env_data[:,0]) #log10 illuminance
#env_data[:,0] = raw_data[1:,1] #illuminance
env_data[:,1] = raw_data[1:,3] #temperature
env_data[:,2] = raw_data[1:,4] #humidity
env_data[:,3] = 1
wfs = raw_data[1:,5:]
wfs_p = wfs[:,20:] #pruned waveforms

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
#linear model. X is env data, Y is sstdr data, M is solved for. 
#Nx92 = Nx4 @ 4x92
#Y = XM
#(X^+)Y = M
#(Yhat) = XM
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

plt.show()

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(x_full[:14601,0],x_full[:14601,1],x_full[:14601,2],label="Old",alpha=0.5)
ax.scatter(x_full[14601:,0],x_full[14601:,1],x_full[14601:,2],label="New",alpha=0.5)
plt.title("New/Old Data Split")
ax.set_xlabel("log10 Illuminance (log10 Lux)")
ax.set_ylabel("Temperature (F)")
ax.set_zlabel("Relative Humidity (%)")
plt.legend()

for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)
    
