#envi_perc.py
#multilayer perceptron to reconstruct SSTDR waveforms from environment data

import torch
import torch.nn as nn
#import matplotlib.pyplot as plt #done below; import procedure differs depending on SHOW_PLOTS
import random
import numpy as np
from tkinter.filedialog import askopenfilename #file selection GUI when loading models
import re #regex for parsing layer strings when loading models

#best NN so far: lr_5e-05_ep_20000_bs_64_L_3_16_32_92_cc_full; log & pruning true

# ======= CONSTANTS ==========
SHOW_PLOTS = True

USE_PAST = True
PAST_SAMPLES = 15 #ignored if USE_PAST is false. past minutes = 2*PAST_SAMPLES
HIDDEN_DIM = 10
N_LAYERS = 3

EPOCHS = 30000
FULL_BATCH = True
BATCH_SIZE = 100 #overridden if FULL_BATCH
LEARNING_RATE = 0.00001
TRAIN_RATIO = 0.75 #ratio of data that is used for training (vs testing)
TEST_RATIO = 1-TRAIN_RATIO
PRINT_PERIOD = 1000 #every X batches we print an update w/ loss & epoch number

LOG = True          #should we take the log10 illuminance value
NORMALIZED = True  #should we normalized the training SSTDR waveforms
PRUNED = False       #should we prune the static beginning portions of the training waveforms

prop_str = "log_"+str(LOG)[0]+"_norm_"+str(NORMALIZED)[0]+"_prune_"+str(PRUNED)[0] #string combining all the above properties to uniquely identify model & results
param_str = "lr_"+str(LEARNING_RATE)+"_ep_"+str(EPOCHS)+"_bs_"+str(BATCH_SIZE)+"_ps_"+("0" if not USE_PAST else str(PAST_SAMPLES))

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
#layer_str = "L_3_24_32_64"+str(WF_SIZE) #string uniquely describing network layout, used to uniquely catalog results
class LinearNet(nn.Module):
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

layer_str = "RNN_L{:d}_H{:d}".format(N_LAYERS,HIDDEN_DIM) #string uniquely describing network layout, used to uniquely catalog results
class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN,self).__init__() #call base class init
        #store some params for later
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        #define layers
        self.rnn = nn.RNN(input_size,hidden_dim,n_layers,batch_first=True)
        self.fc = nn.Linear(hidden_dim,output_size)
        
    def forward(self,x):
        #initialize hidden layer
        batch_size = x.size(0)
        hidden = torch.zeros(self.n_layers,batch_size,self.hidden_dim).double().cuda()
        #obtain outputs
        out, hidden = self.rnn(x,hidden)
        out = out.contiguous().view(-1,self.hidden_dim)
        out = self.fc(out)
        
        return out, hidden

# ======== UNIQUELY IDENTIFY THIS RUN ====

desc_str = param_str+"_"+layer_str+"_"+prop_str #string combining ALL properties: hyperparameters, network, preprocessing methods. used for cataloging results

# ======= LOAD DATA ==========
in_path = "combined_data_new.csv"
raw_data = np.genfromtxt(in_path,delimiter=',',skip_header=1)
times = raw_data[1:,0] #timestamps of measurements
N_env = len(times) #all measurements
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
#for each measurement, include the last PAST_SAMPLES measurements as well. all measurements form a "set".
#we get PAST_SAMPLES*3 fewer valid sets of measurements than we have singular measurements:
#   we have two datasets. the january dataset has data taken once per minute.
#   the july dataset has data taken once per two minutes.
#   we lose PAST_SAMPLES sets in the july dataset, simply because the first PAST_SAMPLES measurements don't have enough previous measurements to form a proper input.
#   we would normally lose another PAST_SAMPLES sets in the january dataset, but we have to increment by 2 instead of one to make sure the data in each set goes back in time at the same rate. thus, we lose PAST_SAMPLES*2 sets.
#   this totals to PAST_SAMPLES*3 fewer sets.

#split data by season, and trim invalid sets (with not enough past data)
N_winter = 14601
N_summer = N_env-N_winter

valid_winter_indices = np.arange(2*PAST_SAMPLES,N_winter)
valid_summer_indices = np.arange(N_winter+PAST_SAMPLES,N_env)
valid_indices = np.concatenate((valid_winter_indices, valid_summer_indices))

winter_test_segment = np.intersect1d(np.arange(1625,1625+int(N_winter*TEST_RATIO)), valid_winter_indices) #manually split testing/training data to give each set a similar distribution
summer_test_segment = np.intersect1d(np.arange(36359-int(N_summer*TEST_RATIO),36359), valid_summer_indices)

test_indices = np.concatenate((winter_test_segment,summer_test_segment))
train_indices = np.setdiff1d(valid_indices,test_indices,assume_unique=True)

#populate X: environment data matrix
N_valid = len(valid_indices)
assert(N_valid == N_env-PAST_SAMPLES*3) #number of valid measurement sets (see above explanation)
X_full = np.zeros((N_env, PAST_SAMPLES, ENV_SIZE)) #X_full contains invalid inputs
for i in range(PAST_SAMPLES*2,N_env):
    if i < N_winter: #january dataset; data taken once per minute
        for j in range(PAST_SAMPLES):
            X_full[i,j,:] = env_data[i-2*PAST_SAMPLES+2*j] #this indexing places the oldest measurement at index 0: it is first in the input sequence
    elif i >= N_winter+PAST_SAMPLES:
        for j in range(PAST_SAMPLES):
            X_full[i,j,:] = env_data[i-PAST_SAMPLES+j]

#split data up based on training/testing sets
valid_timestamps = times[valid_indices]
train_timestamps = times[train_indices]
test_timestamps = times[test_indices]

X_train = X_full[train_indices,:,:]
X_test = X_full[test_indices,:,:]

y_full = wfs #alias for legacy reasons
y_valid = wfs[valid_indices]#np.concatenate((wfs[PAST_SAMPLES*2:N_winter,:], wfs[N_winter+PAST_SAMPLES:,:]))
y_train = wfs[train_indices]
y_test = wfs[test_indices]

N_train = len(X_train)
N_test = len(X_test)
train_indices_s = train_indices.copy() #used for shuffling batches later

assert(len(y_full) == len(X_full))

# ========= TRAINING ===========
def train():
    global BATCH_SIZE
    network = RNN(ENV_SIZE, WF_SIZE, HIDDEN_DIM, N_LAYERS) #instantiate network
    network.double().cuda()
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
        random.shuffle(train_indices_s)
        X_train_s = X_full[train_indices_s]
        y_train_s = y_full[train_indices_s]
        for b in range(batches):
            #for each batch
            #get batch of input data
            b_data = X_train_s[b*BATCH_SIZE:min(N_train, (b+1)*BATCH_SIZE)]
            b_x = torch.from_numpy(b_data).view(-1,PAST_SAMPLES,ENV_SIZE).double().cuda() #batch_size by 3 tensor
            
            #get batch of desired data
            b_desired = y_train_s[b*BATCH_SIZE:min(N_train, (b+1)*BATCH_SIZE)]
            b_y = torch.from_numpy(b_desired).view(-1,WF_SIZE).double().cuda() #batch size by 92 tensor
            
            #predict
            #for p in range(PAST_SAMPLES):        
            #    predictions, hidden = network(b_x[-p]) #only keep the last prediction, which uses the most recent measurement
            predictions, hidden = network(b_x) #indexed by: index in batch, index in sequence, index in measurement
            final_predictions = predictions.view(-1,PAST_SAMPLES,WF_SIZE).double().cuda()[:,-1,:] #take final prediction from sequence TODO: inputs may be going in in reverse order?
            #update weights, record loss
            loss = loss_func(final_predictions, b_y).double().cuda()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_val = loss.cpu().data.numpy().item()
            losses[l_i] = loss_val
            l_i+=1
            
            #print learning information
            if b%PRINT_PERIOD == 0:
                print('Epoch: ' + str(epoch) + ', loss: %.5f' % loss_val)

    #loss curve from training
    plt.figure()
    plt.plot(np.log10(losses))
    plt.title("Training Loss\n"+desc_str)
    plt.xlabel("Iteration")
    plt.ylabel("Log10 MSE Loss")
    plt.tight_layout()
    plt.savefig("plots/"+desc_str+"_loss.png")

    #show all plots
    if SHOW_PLOTS:
        plt.show()
    
    #======= SAVE MODEL STATE ==
    torch.save(network.state_dict(), "models/"+desc_str+"_state_dict")
    return network, losses
    

#======= TESTING ===========
def test(net = None):
    if net is None:
        print("ERROR in test(): received null network object")
        return
    #calculate training/test correlation coeffients
    X_full_tensor = torch.from_numpy(X_full).view(-1,PAST_SAMPLES,ENV_SIZE).double().cuda()
    predictions, hidden = net(X_full_tensor)
    final_predictions = predictions.view(-1,PAST_SAMPLES,WF_SIZE)[:,-1,:]
    results = final_predictions.cpu().detach().numpy()
    p_cc = np.zeros(N_env)
    for i in range(N_env):
        p_cc[i] = np.corrcoef(y_full[i], results[i])[0,1]
        
    #full correlation plot with illuminance
    lw=1
    fig,ax1 = plt.subplots()
    train_segments = np.split(train_indices,np.where(np.diff(train_indices)!=1)[0]+1)
    for seg in train_segments:
        lt1, = ax1.plot(seg,p_cc[seg], label="Training CC",color="C0",lw=lw)
    test_segments = np.split(test_indices,np.where(np.diff(test_indices)!=1)[0]+1)
    for seg in test_segments:
        lt2, = ax1.plot(seg, p_cc[seg], label="Testing CC",color="C1",lw=lw)
    ylims = (0,1)
    ax1.set_ylim(ylims)
    norm_illuminance = np.array(env_data[:,0]) #TODO use train_indices and test_indices here??
    ax2 = ax1.twinx()
    lill, = ax2.plot(norm_illuminance,label="Illuminance",alpha=0.3,lw=1,color="C2")
    ax2.fill_between(range(N_env),norm_illuminance,np.ones(N_env)*np.min(norm_illuminance), alpha=0.2, color="C2")
    ax2.set_title("Prediction Corr. Coeffs\n"+desc_str+'\nAverage C.C.: {:.3f}'.format(np.mean(p_cc[valid_indices])))
    ax1.set_ylabel("CC")
    ax2.set_ylabel("Log10 Illuminance (log Lux)")
    ax2.set_xlabel("Sample")
    plt.legend((lt1,lt2,lill), ("Training CC", "Testing CC", "Illuminance"),loc="lower left")
    plt.tight_layout()
    plt.savefig("plots/"+desc_str+"_cc_full.png")

    #full correlation plot with illuminance, ZOOMED ON CORRELATION
    lw=1
    fig,ax1 = plt.subplots()
    train_segments = np.split(train_indices,np.where(np.diff(train_indices)!=1)[0]+1)
    for seg in train_segments:
        lt1, = ax1.plot(seg,p_cc[seg], label="Training CC",color="C0",lw=lw)
    test_segments = np.split(test_indices,np.where(np.diff(test_indices)!=1)[0]+1)
    for seg in test_segments:
        lt2, = ax1.plot(seg, p_cc[seg], label="Testing CC",color="C1",lw=lw)
    print("Zoomed CC Y limits: "+str(ax1.get_ylim()))
    norm_illuminance = np.array(env_data[:,0]) #TODO use train_indices and test_indices here??
    ax2 = ax1.twinx()
    lill, = ax2.plot(norm_illuminance,label="Illuminance",alpha=0.3,lw=1,color="C2")
    ax2.fill_between(range(N_env),norm_illuminance,np.ones(N_env)*np.min(norm_illuminance), alpha=0.2, color="C2")
    ax1.set_title("LSTM Prediction CC\nTrain mean CC: {:.4f}\n Test mean CC: {:.4f}".format(np.mean(p_cc[train_indices]),np.mean(p_cc[test_indices])))
    ax1.set_ylabel("CC")
    ax2.set_ylabel("Log10 Illuminance (log Lux)")
    ax2.set_xlabel("Sample")
    plt.legend((lt1,lt2,lill), ("Training CC", "Testing CC", "Illuminance"),loc="lower left")
    plt.tight_layout()
    plt.savefig("plots/"+desc_str+"_cc_full_zoomed.png")

    #loss curve from training
    #plt.figure()
    #plt.plot(np.log10(losses))
    #plt.title("Training Loss\n"+desc_str)
    #plt.xlabel("Iteration")
    #plt.ylabel("Log10 MSE Loss")
    #plt.tight_layout()
    #plt.savefig("plots/"+desc_str+"_loss.png")
    
    #plot best pair of waveforms side by side
    plt.figure()
    max_test_i = np.argmax(p_cc[test_indices])
    max_i = test_indices[max_test_i]
    env = X_full[max_i,-1,:]
    plt.plot(y_full[max_i,:], label="Measured")
    plt.plot(results[max_i,:], label="Predicted")
    plt.legend()
    plt.ylabel("Magnitude")
    plt.xlabel("Sample")
    format_str = "i={:d}\nCC: {:.4f}\nIll.: {:.3f}, degF: {:.2f}, RH: {:.2f}\nt: {:f}"
    plt.title("Best Testing Set Pair, "+format_str.format(max_i,p_cc[max_i],env[0],env[1],env[2],times[max_i]))
    plt.tight_layout()

    #plot worst pair of waveforms side by side
    plt.figure()
    min_test_i = np.argmin(p_cc[test_indices])
    min_i = test_indices[min_test_i]
    env = X_full[min_i,-1,:]
    plt.plot(y_full[min_i,:], label="Measured")
    plt.plot(results[min_i,:], label="Predicted")
    plt.legend()
    plt.ylabel("Magnitude")
    plt.xlabel("Sample")
    plt.title("Worst Testing Set Pair, "+format_str.format(min_i,p_cc[min_i],env[0],env[1],env[2],times[min_i]))
    plt.tight_layout()
    
    #show all plots
    if SHOW_PLOTS:
        plt.show()

def load_model(path = None):
    if path is None:
        path = askopenfilename() #open OS GUI to locate a saved model dictionary
    if path == '':
        print("ERROR in load_model(): Empty path to state dictionary received.")
        return
    plate = re.findall("L\d+_H\d+",path)[0] #locate the layer count and hidden layer dim based on the state dictionary filename; "plate" is short for nameplate, it's just what I traditionally use in similar regex code
    layers, h_dim = [int(s) for s in re.findall("\d+",plate)] #extract the numbers themselves from this string
    network = RNN(ENV_SIZE,WF_SIZE,h_dim, layers) #instantiate a properly sized network object
    network.load_state_dict(torch.load(path))
    network.double().cuda()
    return network
    
def exec_model(network):
    if network is None:
        print("ERROR in exec_model: Received None instead of network object")
        return
    #input to tensor, run through, output to numpy, return input & output as numpy
    X_full_tensor = torch.from_numpy(X_full).view(-1,PAST_SAMPLES,ENV_SIZE).double().cuda()
    predictions, hidden = network(X_full_tensor)
    final_predictions = predictions.view(-1,PAST_SAMPLES,WF_SIZE)[:,-1,:]
    results = final_predictions.cpu().detach().numpy()
    return results

        

#==========================
if __name__ == '__main__':
    net,loss = train()
    test(net)