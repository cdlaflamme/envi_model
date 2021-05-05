#envi_perc.py
#multilayer perceptron to reconstruct SSTDR waveforms from environment data

import torch
import torch.nn as nn
#import matplotlib.pyplot as plt #done below; import procedure differs depending on SHOW_PLOTS
import random
import numpy as np
from tkinter.filedialog import askopenfilename #file selection GUI when loading models
import re #regex for parsing layer strings when loading models
from scheduler import *
import traceback
#best RNN so far: lr_5e-05_ep_20000_bs_64_L_3_16_32_92_cc_full; log & pruning true

# ======= CONSTANTS ==========
SHOW_PLOTS = True

PAST_SAMPLES = 10 #past minutes = 2*PAST_SAMPLES
HIDDEN_DIM = 10
N_LAYERS = 3

EPOCHS = 1000
FULL_BATCH = False
BATCH_SIZE = 300 #overridden if FULL_BATCH
LEARNING_RATE = 2e-6
DECAY = 0.5
DECAY_MODE = Schedule.EXP
EPOCHS_PER_DECAY = 10 #used in step mode
WARMUP_EPOCHS = 3

TRAIN_RATIO = 0.75 #ratio of data that is used for training (vs testing)
TEST_RATIO = 1-TRAIN_RATIO
PRINT_PERIOD = 100 #every X batches we print an update w/ loss & epoch number

LSTM = False #if false, will use a basic RNN
LOG = True          #should we take the log10 illuminance value
NORMALIZED = True  #should we normalized the training SSTDR waveforms
PRUNED = False       #should we prune the static beginning portions of the training waveforms
#OUT_ACTIVATION = True #should there be an activation function at the output (only makes sense if normalizing. Uses TanH because output is in [-1,1])

# ======= SECONDARY CONSTANTS ======

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

# ======= UNIQUE STRING REPRESENTATION OF CONFIGURATION ======

LAYER_STR = "RNN_L{:d}_H{:d}".format(N_LAYERS,HIDDEN_DIM)
PROP_STR = "log_"+str(LOG)[0]+"_norm_"+str(NORMALIZED)[0]+"_prune_"+str(PRUNED)[0]+"_lstm_"+str(LSTM)[0]
PARAM_STR = "ep_"+str(EPOCHS)+"_bs_"+str(BATCH_SIZE)+"_ps_"+str(PAST_SAMPLES)
LEARNING_STR = "lr_"+str(LEARNING_RATE)+"_dk_"+str(DECAY)+"_mode_"+DECAY_MODE.name+"_epd_"+str(EPOCHS_PER_DECAY)+"_wue_"+str(WARMUP_EPOCHS)

DESC_STR = LAYER_STR+"_"+PROP_STR+"_"+PARAM_STR+"_"+LEARNING_STR

# ======== NETWORK DEFINITION ====
class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super().__init__() #call base class init
        #store some params for later
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        #define layers
        if LSTM:
            self.rnn = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)
        else:
            self.rnn =  nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
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

# ======= LOAD DATA ==========
print("Loading data...")
in_path = "combined_data.csv"
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
N_winter = 14595
N_summer = N_env-N_winter

valid_winter_indices = np.arange(2*PAST_SAMPLES,N_winter)
valid_summer_indices = np.arange(N_winter+PAST_SAMPLES,N_env)
valid_indices = np.concatenate((valid_winter_indices, valid_summer_indices))

#manually split testing/training data to give each set a similar distribution
winter_test_segment = np.intersect1d(np.arange(1625,1625+int(N_winter*TEST_RATIO)), valid_winter_indices) 
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
    print("Training...")
    try:
        global BATCH_SIZE
        #count batches
        if FULL_BATCH:
            n_batches = 1
            BATCH_SIZE = N_train
        else:
            n_batches = int(N_train/BATCH_SIZE)
        
        #tensorize data
        X_full_tensor = torch.from_numpy(X_full).view(-1,PAST_SAMPLES,ENV_SIZE).double().cuda()
        y_full_tensor = torch.from_numpy(y_full).view(-1,WF_SIZE).double().cuda()
        #instantiate network
        network = Network(ENV_SIZE, WF_SIZE, HIDDEN_DIM, N_LAYERS)
        network.double().cuda()
        #infrastructure
        optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE) #use ADAM for optimization
        scheduler = Scheduler(WF_SIZE, 1, WARMUP_EPOCHS*n_batches, optimizer, mode=Schedule.EXP, exp_decay=DECAY, step_count=EPOCHS_PER_DECAY*n_batches, static_lr=LEARNING_RATE)
        loss_func = nn.MSELoss() #use MSE objective function
        
        losses = np.zeros(EPOCHS * n_batches) #create empty vector for tracking loss
        learning_rates = np.zeros(EPOCHS * n_batches) #create empty vector for tracking learning rate
        step = 0 #crude indexing variable for loss/LR vectors (stored per step, not per epoch)
        for epoch in range(EPOCHS):
            print("Running epoch {0}".format(epoch))
            #shuffle order of training data
            #indexed by: index in batch, index in sequence, index in measurement
            random.shuffle(train_indices_s)
            X_train_s = X_full_tensor[train_indices_s]
            y_train_s = y_full_tensor[train_indices_s]
            for b in range(n_batches):
                optimizer.zero_grad()
                #for each batch
                #get batch of input data
                b_x = X_train_s[b*BATCH_SIZE:min(N_train, (b+1)*BATCH_SIZE)]
                #get batch of desired data
                b_y = y_train_s[b*BATCH_SIZE:min(N_train, (b+1)*BATCH_SIZE)]
                #predict
                predictions, hidden = network(b_x)
                #take final prediction from output sequences
                final_predictions = predictions.view(-1,PAST_SAMPLES,WF_SIZE).double().cuda()[:,-1,:] 
                #calculate loss, backpropagate
                loss = loss_func(final_predictions, b_y).double().cuda()
                loss.backward()
                scheduler.step()
                #store info
                step_rate = scheduler.rate()
                loss_val = loss.data.item()
                losses[step] = loss_val
                learning_rates[step] = step_rate
                step+=1
                
                #print learning information
                if b==0 or b%PRINT_PERIOD == 0:
                    print('\tbatch {:}: loss: {:.5f}; LR: {:.2e}'.format(b, loss_val, step_rate))
        #loss curve from training, by step
        plt.figure()
        loss_plot, = plt.plot(np.log10(losses))
        plt.title("Training Loss, by Step")
        plt.xlabel("Step [Batch]")
        plt.ylabel("Log10 MSE Loss")
        plt.twinx()
        lr_plot, = plt.plot(learning_rates, color="C1", lw=1)
        plt.ylabel("Learning Rate")
        plt.legend((loss_plot, lr_plot), ("Loss", "Learning Rate"), loc='upper right')
        plt.tight_layout()
        plt.savefig("plots/"+DESC_STR+"_step_loss.png")
        
        #loss curve from training, by epoch
        plt.figure()
        step_x = np.linspace(0,EPOCHS-1,len(losses))
        epoch_loss = np.mean(losses.reshape((EPOCHS, -1)), axis=1)
        loss_plot, = plt.plot(np.log10(epoch_loss))
        plt.title("Training Loss, by Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Log10 MSE Loss")
        plt.twinx()
        lr_plot, = plt.plot(step_x, learning_rates, color="C1", lw=1)
        plt.ylabel("Learning Rate")
        plt.legend((loss_plot, lr_plot), ("Loss", "Learning Rate"), loc='upper right')
        plt.tight_layout()
        plt.savefig("plots/"+DESC_STR+"_epoch_loss.png")
        
    except:
        traceback.print_exc()
    finally:    
        #======= SAVE MODEL ======
        network.desc_str = DESC_STR
        network.train_losses = losses
        network.learning_rates = learning_rates
        torch.save(network, "models/"+DESC_STR+".sav")
        #show plots
        if SHOW_PLOTS:
            plt.show()
        return network

#======= TESTING ===========
def test(net = None):
    print("Evaluating model...")
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
    #ax2.set_title("Prediction Corr. Coeffs\n"+DESC_STR+'\nAverage CC: {:.3f}'.format(np.mean(p_cc[valid_indices])))
    ax2.set_title('Prediction Corr. Coeffs\nAverage CC: {:.3f}'.format(np.mean(p_cc[valid_indices])))
    ax1.set_ylabel("CC")
    ax2.set_ylabel("Log10 Illuminance (log Lux)")
    ax2.set_xlabel("Sample")
    plt.legend((lt1,lt2,lill), ("Training CC", "Testing CC", "Illuminance"),loc="lower left")
    plt.tight_layout()
    plt.savefig("plots/"+DESC_STR+"_cc_full.png")

    #full correlation plot with illuminance, ZOOMED ON CORRELATION
    lw=1
    fig,ax1 = plt.subplots()
    train_segments = np.split(train_indices,np.where(np.diff(train_indices)!=1)[0]+1)
    for seg in train_segments:
        lt1, = ax1.plot(seg,p_cc[seg], label="Training CC",color="C0",lw=lw)
    test_segments = np.split(test_indices,np.where(np.diff(test_indices)!=1)[0]+1)
    for seg in test_segments:
        lt2, = ax1.plot(seg, p_cc[seg], label="Testing CC",color="C1",lw=lw)
    #print("Zoomed CC Y limits: "+str(ax1.get_ylim()))
    norm_illuminance = np.array(env_data[:,0]) #TODO use train_indices and test_indices here??
    ax2 = ax1.twinx()
    lill, = ax2.plot(norm_illuminance,label="Illuminance",alpha=0.3,lw=1,color="C2")
    ax2.fill_between(range(N_env),norm_illuminance,np.ones(N_env)*np.min(norm_illuminance), alpha=0.2, color="C2")
    ax1.set_title("LSTM Prediction CC\nTrain mean CC: {:.4f}\n Test mean CC: {:.4f}".format(np.mean(p_cc[train_indices]),np.mean(p_cc[test_indices])))
    ax1.set_ylabel("CC")
    ax2.set_ylabel("Log10 Illuminance (log Lux)")
    ax1.set_xlabel("Sample")
    plt.legend((lt1,lt2,lill), ("Training CC", "Testing CC", "Illuminance"),loc="lower left")
    plt.tight_layout()
    plt.savefig("plots/"+DESC_STR+"_cc_full_zoomed.png")

    #loss curve from training
    #plt.figure()
    #plt.plot(np.log10(losses))
    #plt.title("Training Loss")
    #plt.xlabel("Iteration")
    #plt.ylabel("Log10 MSE Loss")
    #plt.tight_layout()
    #plt.savefig("plots/"+DESC_STR+"_loss.png")
    
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
    plt.savefig("plots/"+DESC_STR+"best_pair.png")

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
    plt.savefig("plots/"+DESC_STR+"worst_pair.png")
    
    #show all plots
    if SHOW_PLOTS:
        plt.show()

def load_model(path = None):
    if path is None:
        path = askopenfilename() #open OS GUI to locate a saved model dictionary
    if path == '':
        print("ERROR in load_model(): Empty path to state dictionary received.")
        return
    return torch.load(path)
    """
    plate = re.findall("L\d+_H\d+",path)[0] #locate the layer count and hidden layer dim based on the state dictionary filename; "plate" is short for nameplate, it's just what I traditionally use in similar regex code
    layers, h_dim = [int(s) for s in re.findall("\d+",plate)] #extract the numbers themselves from this string
    network = RNN(ENV_SIZE,WF_SIZE,h_dim, layers) #instantiate a properly sized network object
    network.load_state_dict(torch.load(path))
    network.double().cuda()
    return network
    """
    
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
    net = train()
    test(net)