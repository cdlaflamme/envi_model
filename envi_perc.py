#envi_perc.py
#multilayer perceptron to reconstruct SSTDR waveforms from environment data

import torch
import torch.nn as nn
#import matplotlib.pyplot as plt #done below; import procedure differs depending on SHOW_PLOTS
import random
import numpy as np

#best NN so far: lr_5e-05_ep_20000_bs_64_L_3_16_32_92_cc_full; log & pruning true

# ======= CONSTANTS ==========
SHOW_PLOTS = True

EPOCHS = 60000
FULL_BATCH = True
BATCH_SIZE = 100 #overridden if FULL_BATCH
LEARNING_RATE = 0.000005
TRAIN_RATIO = 0.75 #ratio of data that is used for training (vs testing)
PRINT_PERIOD = 1000 #every X batches we print an update w/ loss & epoch number

LOG = True          #should we take the log10 illuminance value
NORMALIZED = True  #should we normalized the training SSTDR waveforms
PRUNED = False       #should we prune the static beginning portions of the training waveforms

prop_str = "log_"+str(LOG)[0]+"_norm_"+str(NORMALIZED)[0]+"_prune_"+str(PRUNED)[0] #string combining all the above properties to uniquely identify model & results
param_str = "lr_"+str(LEARNING_RATE)+"_ep_"+str(EPOCHS)+"_bs_"+str(BATCH_SIZE)

if PRUNED:
    WF_SIZE = 92-20
else:
    WF_SIZE = 92

ENV_SIZE = 3 #3 samples of environment: illuminance, temperature, humidity

if SHOW_PLOTS:
    import matplotlib.pyplot as plt
else:
    import matplotlib
    matplotlib.use('Agg') #this will make everything work even when there is no X server
    import matplotlib.pyplot as plt 

# ======== NETWORK DEFINITION ====
layer_str = "L_3_24_32_64"+str(WF_SIZE) #string uniquely describing network layout, used to uniquely catalog results
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(ENV_SIZE,24),
            nn.ReLU(),
            nn.Linear(24,32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,WF_SIZE)
        )
    
    def forward(self,x):
        results = self.net(x)
        return results

# ======== UNIQUELY IDENTIFY THIS RUN ====

desc_str = param_str+"_"+layer_str+"_"+prop_str #string combining ALL properties: hyperparameters, network, preprocessing methods. used for cataloging results

# ======= LOAD DATA ==========
in_path = "combined_data_new.csv"
raw_data = np.genfromtxt(in_path,delimiter=',',skip_header=1)
times = raw_data[1:,0] #timestamps of measurements
N_env = len(times)
env_data = np.zeros((N_env,ENV_SIZE)) #to be filled from the raw data file. values are read from specific columns as determined by our sensor (which generates the files)
if LOG:
    env_data[:,0] = np.where(raw_data[1:,1]==0, 0.01, raw_data[1:,1]) #special case for 0 lux
    env_data[:,0] = np.log10(env_data[:,0]) #log10 illuminance
else:
    env_data[:,0] = raw_data[1:,1] #illuminance
env_data[:,1] = raw_data[1:,3] #temperature
env_data[:,2] = raw_data[1:,4] #humidity
if ENV_SIZE > 3:
    env_data[:,3] = 1
wfs = raw_data[1:,5:]

if NORMALIZED:
    maxes = np.max(wfs,axis=1)
    wfs = np.array([wfs[i]/maxes[i] for i in range(N_env)])
if PRUNED:
    wfs = wfs[:,20:]
    
#============ SPLIT DATA ==============
x_full = env_data
border = int(len(x_full)*TRAIN_RATIO) #cutoff index separating training and test data
x_train = x_full[0:border]
x_test = x_full[border:]

y_full = wfs
y_train = y_full[0:border]
y_test = y_full[border:]

N_train = len(x_train)
N_test = len(x_test)
train_indices = list(range(N_train)) #used for shuffling batches later

# ========= TRAINING ===========
network = Net().double() #instantiate network 
optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE) #use ADAM for optimization
loss_func = nn.MSELoss() #use MSE objective function
if FULL_BATCH:
    batches = 1
    BATCH_SIZE = N_train
else:
    batches = int(N_train/BATCH_SIZE)
losses = np.zeros(EPOCHS * batches) #create empty vector for tracking loss over time (for generating a learning curve)
l_i = 0 #crude indexing variable for loss vector
for epoch in range(EPOCHS):
    #shuffle order of training data
    random.shuffle(train_indices)
    x_train_s = np.array([x_train[i] for i in train_indices])
    y_train_s = np.array([y_train[i] for i in train_indices])
    for b in range(batches):
        #for each batch
        #get batch of input data
        b_data = x_train_s[b*BATCH_SIZE:min(N_train, (b+1)*BATCH_SIZE)]
        b_x = torch.from_numpy(b_data).view(-1,ENV_SIZE).double() #batch_size by 3 tensor
        
        #get batch of desired data
        b_desired = y_train_s[b*BATCH_SIZE:min(N_train, (b+1)*BATCH_SIZE)]
        b_y = torch.from_numpy(b_desired).view(-1,WF_SIZE).double() #batch size by 92 tensor
        
        #predict
        predictions = network(b_x)
        
        #update weights, record loss
        loss = loss_func(predictions, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses[l_i] = loss.data.numpy()
        l_i+=1
        
        #print learning information
        if b%PRINT_PERIOD == 0:
            print('Epoch: ' + str(epoch) + ', loss: %.4f' % loss.data.numpy())

#======= SAVE MODEL STATE ==
torch.save(network.state_dict(), "models/"+desc_str+"_state_dict")

#======= TESTING ===========
"""
#correlation plot: correlating predicted waveforms with real waveforms
plt.figure()
x_test_tensor = torch.from_numpy(x_test).view(-1,ENV_SIZE).double()
test_results = network(x_test_tensor).detach().numpy()
cc_test = np.zeros(N_test)
for i in range(N_test):
    cc_test[i] = np.corrcoef(y_test[i], test_results[i])[0,1]
plt.plot(cc_test)
plt.ylim((0,1))
plt.title("Testing Set Prediction Corr. Coeffs\n"+desc_str)
plt.ylabel("CC")
plt.xlabel("Sample")
plt.savefig("plots/cc_test_"+desc_str)

#correlation plot from training
plt.figure()
x_train_tensor = torch.from_numpy(x_train).view(-1,ENV_SIZE).double()
train_results = network(x_train_tensor).detach().numpy()
cc_train = np.zeros(N_train)
for i in range(N_train):
    cc_train[i] = np.corrcoef(y_train[i], train_results[i])[0,1]
plt.plot(cc_train)
plt.ylim((0,1))
plt.title("Training Set Prediction Corr. Coeffs\n"+desc_str)
plt.ylabel("CC")
plt.xlabel("Sample")
plt.savefig("plots/cc_train_"+desc_str)
"""

#full correlation plot with illuminance
plt.figure()
x_test_tensor = torch.from_numpy(x_test).view(-1,ENV_SIZE).double()
test_results = network(x_test_tensor).detach().numpy()
cc_test = np.zeros(N_test)
for i in range(N_test):
    cc_test[i] = np.corrcoef(y_test[i], test_results[i])[0,1]
x_train_tensor = torch.from_numpy(x_train).view(-1,ENV_SIZE).double()
train_results = network(x_train_tensor).detach().numpy()
cc_train = np.zeros(N_train)
for i in range(N_train):
    cc_train[i] = np.corrcoef(y_train[i], train_results[i])[0,1]
norm_illuminance = np.array(env_data[:,0])
norm_illuminance = norm_illuminance/max(norm_illuminance)
plt.plot(np.arange(N_train), cc_train,label="Training CC")
plt.plot(np.arange(N_train, N_test+N_train), cc_test,label="Testing CC")
plt.plot(norm_illuminance,label="Illuminance",alpha=0.5)
plt.ylim((0,1))
plt.title("Prediction Corr. Coeffs\n"+desc_str+'\nAverage C.C.: {:.3f}'.format(np.mean(np.concatenate((cc_test,cc_train)))))
plt.ylabel("CC")
plt.xlabel("Sample")
plt.legend()
plt.tight_layout()
plt.savefig("plots/"+desc_str+"_cc_full")

#full correlation plot with illuminance, ZOOMED ON CORRELATION
plt.figure()
#x_test_tensor = torch.from_numpy(x_test).view(-1,ENV_SIZE).double()
#test_results = network(x_test_tensor).detach().numpy()
#cc_test = np.zeros(N_test)
#for i in range(N_test):
#    cc_test[i] = np.corrcoef(y_test[i], test_results[i])[0,1]
#x_train_tensor = torch.from_numpy(x_train).view(-1,ENV_SIZE).double()
#train_results = network(x_train_tensor).detach().numpy()
#cc_train = np.zeros(N_train)
#for i in range(N_train):
#    cc_train[i] = np.corrcoef(y_train[i], train_results[i])[0,1]
plt.plot(np.arange(N_train), cc_train,label="Training CC")
plt.plot(np.arange(N_train, N_test+N_train), cc_test,label="Testing CC")
ylims = plt.ylim()
norm_illuminance = np.array(env_data[:,0])
norm_illuminance = norm_illuminance/max(norm_illuminance)*(ylims[1]-ylims[0])+ylims[0]
plt.plot(norm_illuminance,label="Illuminance",alpha=0.5)
plt.title("Zoomed Prediction Corr. Coeffs\n"+desc_str+'\nAverage C.C.: {:.3f}'.format(np.mean(np.concatenate((cc_test,cc_train)))))
plt.ylabel("CC")
plt.xlabel("Sample")
plt.legend()
plt.tight_layout()
plt.savefig("plots/"+desc_str+"_cc_full_zoomed")


#loss curve from training
plt.figure()
plt.plot(np.log10(losses))
plt.title("Training Loss\n"+desc_str)
plt.xlabel("Iteration")
plt.ylabel("Log10 MSE Loss")
plt.savefig("plots/"+desc_str+"_loss")

#show all plots
if SHOW_PLOTS:
    plt.show()
