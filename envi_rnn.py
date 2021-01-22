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

EPOCHS = 50000
FULL_BATCH = False
BATCH_SIZE = 100 #overridden if FULL_BATCH
LEARNING_RATE = 0.00005
TRAIN_RATIO = 0.75 #ratio of data that is used for training (vs testing)
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
N_full = N_env-PAST_SAMPLES*3 #number of valid measurement sets (see above explanation)
N_january = 14601 #measurements before this index are from january
X_full = np.zeros((N_full, PAST_SAMPLES, ENV_SIZE))
x_i = 0
for i in range(PAST_SAMPLES*2,N_env):
    if i < N_january: #january dataset; data taken once per minute
        for j in range(PAST_SAMPLES):
            X_full[x_i,j,:] = env_data[i-2*PAST_SAMPLES+2*j] #this indexing places the oldest measurement at index 0: it is first in the input sequence
        x_i +=1
    elif i >= N_january+PAST_SAMPLES:
        for j in range(PAST_SAMPLES):
            X_full[x_i,j,:] = env_data[i-PAST_SAMPLES+j]
        x_i +=1
border = int(N_full*TRAIN_RATIO) #cutoff index separating training and test data

valid_timestamps = np.concatenate((times[PAST_SAMPLES*2:N_january], times[N_january+PAST_SAMPLES:]))
train_timestamps = valid_timestamps[0:border]
test_timestamps = valid_timestamps[border:]

X_train = X_full[0:border,:,:]
X_test = X_full[border:,:,:]

y_full = np.concatenate((wfs[PAST_SAMPLES*2:N_january,:], wfs[N_january+PAST_SAMPLES:,:]))
y_train = y_full[0:border]
y_test = y_full[border:]

N_train = len(X_train)
N_test = len(X_test)
train_indices = list(range(N_train)) #used for shuffling batches later

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
        random.shuffle(train_indices)
        X_train_s = np.array([X_train[i] for i in train_indices])
        y_train_s = np.array([y_train[i] for i in train_indices])
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
    X_test_tensor = torch.from_numpy(X_test).view(-1,PAST_SAMPLES,ENV_SIZE).double().cuda()
    #for p in range(PAST_SAMPLES):
    #    predictions, hidden = net(X_test_tensor)
    predictions, hidden = net(X_test_tensor)
    final_predictions = predictions.view(-1,PAST_SAMPLES,WF_SIZE)[:,-1,:]
    test_results = final_predictions.cpu().detach().numpy()
    cc_test = np.zeros(N_test)
    for i in range(N_test):
        cc_test[i] = np.corrcoef(y_test[i], test_results[i])[0,1]
        
    X_train_tensor = torch.from_numpy(X_train).view(-1,PAST_SAMPLES,ENV_SIZE).double().cuda()
    #for p in range(PAST_SAMPLES):
    #    predictions, hidden = net(X_train_tensor)
    predictions, hidden = net(X_train_tensor)
    final_predictions = predictions.view(-1,PAST_SAMPLES,WF_SIZE)[:,-1,:]
    train_results = final_predictions.cpu().detach().numpy()
    cc_train = np.zeros(N_train)
    for i in range(N_train):
        cc_train[i] = np.corrcoef(y_train[i], train_results[i])[0,1]
        
    #full correlation plot with illuminance
    plt.figure()
    plt.plot(np.arange(N_train), cc_train,label="Training CC")
    plt.plot(np.arange(N_train, N_test+N_train), cc_test,label="Testing CC")
    ylims = (0,1)
    plt.ylim((0,1))
    norm_illuminance = np.array(env_data[:,0]) #TODO use train_indices and test_indices here??
    norm_illuminance = norm_illuminance/max(norm_illuminance)*(ylims[1]-ylims[0])+ylims[0]
    plt.plot(norm_illuminance,label="Illuminance",alpha=0.5)
    plt.fill_between(range(N_env),norm_illuminance,np.ones(N_env)*np.min(norm_illuminance), alpha=0.3, color="C2")
    plt.title("Prediction Corr. Coeffs\n"+desc_str+'\nAverage C.C.: {:.3f}'.format(np.mean(np.concatenate((cc_test,cc_train)))))
    plt.ylabel("CC")
    plt.xlabel("Sample")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/"+desc_str+"_cc_full.png")

    #full correlation plot with illuminance, ZOOMED ON CORRELATION
    plt.figure()
    #X_test_tensor = torch.from_numpy(X_test).view(-1,ENV_SIZE)
    #test_results = net(X_test_tensor).detach().numpy()
    #cc_test = np.zeros(N_test)
    #for i in range(N_test):
    #    cc_test[i] = np.corrcoef(y_test[i], test_results[i])[0,1]
    #X_train_tensor = torch.from_numpy(X_train).view(-1,ENV_SIZE)
    #train_results = net(X_train_tensor).detach().numpy()
    #cc_train = np.zeros(N_train)
    #for i in range(N_train):
    #    cc_train[i] = np.corrcoef(y_train[i], train_results[i])[0,1]
    plt.plot(np.arange(N_train), cc_train,label="Training CC")
    plt.plot(np.arange(N_train, N_test+N_train), cc_test,label="Testing CC")
    ylims = plt.ylim()
    norm_illuminance = np.array(env_data[:,0])
    norm_illuminance = norm_illuminance/max(norm_illuminance)*(ylims[1]-ylims[0])+ylims[0]
    plt.fill_between(range(N_env),norm_illuminance,np.ones(N_env)*np.min(norm_illuminance), alpha=0.3, color="C2")
    plt.plot(norm_illuminance,label="Illuminance",alpha=0.5)
    plt.title("Zoomed Prediction Corr. Coeffs\n"+desc_str+'\nAverage C.C.: {:.3f}'.format(np.mean(np.concatenate((cc_test,cc_train)))))
    plt.ylabel("CC")
    plt.xlabel("Sample")
    plt.legend()
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
    max_i = np.argmax(cc_test)
    env = X_full[max_i,-1,:]
    plt.plot(y_test[max_i,:], label="Measured")
    plt.plot(test_results[max_i,:], label="Predicted")
    plt.legend()
    plt.ylabel("Magnitude")
    plt.xlabel("Sample")
    format_str = "i={:d}\nCC: {:.4f}\nIll.: {:.3f}, degF: {:.2f}, RH: {:.2f}\nt: {:f}"
    plt.title("Best Testing Set Pair, "+format_str.format(max_i,cc_test[max_i],env[0],env[1],env[2],test_timestamps[max_i]))
    plt.tight_layout()
    #TODO change this to plot the predicted wf
    #plot worst pair of waveforms side by side
    plt.figure()
    min_i = np.argmin(cc_test)
    env = X_full[min_i,-1,:]
    plt.plot(y_test[min_i,:], label="Measured")
    plt.plot(test_results[min_i,:], label="Predicted")
    plt.legend()
    plt.ylabel("Magnitude")
    plt.xlabel("Sample")
    plt.title("Worst Testing Set Pair, "+format_str.format(min_i,cc_test[min_i],env[0],env[1],env[2],test_timestamps[min_i]))
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