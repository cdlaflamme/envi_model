#envi_perc.py
#multilayer perceptron to reconstruct SSTDR waveforms from environment data

import torch
import torch.nn as nn
#import matplotlib.pyplot as plt #done below; import procedure differs depending on SHOW_PLOTS
import random
import numpy as np
from tkinter.filedialog import askopenfilename #file selection GUI when loading models
from scipy.interpolate import interp1d


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
FEET_PER_SAMPLE = 3.63716
METERS_PER_FOOT = 0.3048

if SHOW_PLOTS:
    import matplotlib.pyplot as plt
else:
    import matplotlib
    matplotlib.use('Agg') #this will make everything work even when there is no X server
    import matplotlib.pyplot as plt 

# ======== NETWORK DEFINITION ====
layer_str = "L_3_24_32_64_92"+str(WF_SIZE) #string uniquely describing network layout, used to uniquely catalog results
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
    env_data[:,0] = illum = np.log10(env_data[:,0]) #log10 illuminance
else:
    env_data[:,0] = illum = raw_data[1:,1] #illuminance
env_data[:,1] = degF = raw_data[1:,3] #temperature
env_data[:,2] = RH = raw_data[1:,4] #humidity
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

N_full = len(x_full)
N_train = len(x_train)
N_test = len(x_test)
train_indices = list(range(N_train)) #used for shuffling batches later

# ========= TRAINING ===========
def train():
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
    
    #loss curve from training
    plt.figure()
    plt.plot(np.log10(losses))
    plt.title("Training Loss\n"+desc_str)
    plt.xlabel("Iteration")
    plt.ylabel("Log10 MSE Loss")
    plt.savefig("plots/"+desc_str+"_loss")
    
    if SHOW_PLOTS:
        plt.show()
    
    return network
    
#======= TESTING ===========
def test(network):
    x_full_tensor = torch.from_numpy(x_full).view(-1,ENV_SIZE).double()
    p_full = network(x_full_tensor).detach().numpy()
    p_cc = np.zeros(N_full)
    for i in range(N_full):
        p_cc[i] = np.corrcoef(y_full[i], p_full[i])[0,1]
    p_mse = np.mean((p_full-y_full)**2,axis=1)    
    
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
    plt.title("Prediction Corr. Coeffs\n"+desc_str+'\nAverage CC: {:.3f}'.format(np.mean(np.concatenate((cc_train,cc_test)))))
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
    plt.title("Zoomed Prediction Corr. Coeffs\n"+desc_str+'\nTrain mean CC: {:.4f}\n Test mean CC: {:.4f}'.format(np.mean(cc_test),np.mean(cc_train)))
    plt.ylabel("CC")
    plt.xlabel("Sample")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/"+desc_str+"_cc_full_zoomed")

    
    # ==== COPIED FROM LINEAR MODEL ====
    #=========== plotting variation from each factor ====================
    N_simulations = 10000 #number of waveforms to generate for each factor (varied factor sampled uniformly at this many points across measured range)
    INTERP_DATA = False
    INTERP_VISUAL = True
    INTERP_SIZE = 1000
    out_folder = "perc_plots"
    illum_index = 0
    degF_index = 1
    RH_index = 2
    
    
    #find mode daytime illuminance
    illum_counts, illum_centers = np.histogram(illum)
    illum_mode_bin_i = np.argmax(illum_counts)
    illum_mode = illum_centers[illum_mode_bin_i]

    #find mode temperature
    degF_counts, degF_centers = np.histogram(degF)
    degF_mode_bin_i = np.argmax(degF_counts)
    degF_mode = degF_centers[degF_mode_bin_i]

    #find mode humidity
    RH_counts, RH_centers = np.histogram(RH)
    RH_mode_bin_i = np.argmax(RH_counts)
    RH_mode = RH_centers[RH_mode_bin_i]

    #Illuminance: fix temperature and humidity
    illum_I = np.ones((N_simulations,ENV_SIZE))
    illum_I[:,illum_index] = np.linspace(np.min(illum),np.max(illum),N_simulations);
    illum_I[:,degF_index] = degF_mode
    illum_I[:,RH_index] = RH_mode
    illum_I_tensor = torch.from_numpy(illum_I).view(-1,ENV_SIZE).double()
    illum_Y = network(illum_I_tensor).detach().numpy()

    degF_I = np.ones((N_simulations,ENV_SIZE))
    degF_I[:,illum_index] = illum_mode
    degF_I[:,degF_index] = np.linspace(np.min(degF),np.max(degF),N_simulations);
    degF_I[:,RH_index] = RH_mode
    degF_I_tensor = torch.from_numpy(degF_I).view(-1,ENV_SIZE).double()
    degF_Y = network(degF_I_tensor).detach().numpy()

    RH_I = np.ones((N_simulations,ENV_SIZE))
    RH_I[:,illum_index] = illum_mode
    RH_I[:,degF_index] = degF_mode
    RH_I[:,RH_index] = np.linspace(np.min(RH),np.max(RH),N_simulations);
    RH_I_tensor = torch.from_numpy(RH_I).view(-1,ENV_SIZE).double()
    RH_Y = network(RH_I_tensor).detach().numpy()

    if INTERP_VISUAL and not INTERP_DATA:
        x = np.arange(WF_SIZE)
        xx = np.linspace(0,WF_SIZE-1,INTERP_SIZE)
        illum_Y = np.array([interp1d(x,wf,kind='cubic')(xx) for wf in illum_Y])
        degF_Y  = np.array([interp1d(x,wf,kind='cubic')(xx) for wf in degF_Y])
        RH_Y    = np.array([interp1d(x,wf,kind='cubic')(xx) for wf in RH_Y])

    #each plot will share the same code. extra initial investment, but reduces headaches later.
    mode_strings = ("Illuminance: 10^{:.2f} Lux".format(illum_mode), "Temperature: {:.2f} F".format(degF_mode), "Humidity: {:.2f} %".format(RH_mode))
    lw = 1
    for i,(name,Y) in enumerate(zip(("Illuminance","Temperature","Humidity"),(illum_Y,degF_Y,RH_Y))):
        used_m_strings = [m for j,m in enumerate(mode_strings) if i!=j]
        fig,ax1 = plt.subplots()
        ax1.set_title("Variation Due to {:s}\n{:s}\n{:s}".format(name,used_m_strings[0],used_m_strings[1]))
        ax1.set_xlabel("Distance (meters)")
        ax1.set_ylabel("Normalized SSTDR Magnitude")
        std_devs = np.std(Y,axis=0)
        if INTERP_DATA or INTERP_VISUAL:
            meters = np.arange(INTERP_SIZE)*FEET_PER_SAMPLE*WF_SIZE/INTERP_SIZE*METERS_PER_FOOT
        else:
            meters = np.arange(WF_SIZE)*FEET_PER_SAMPLE*METERS_PER_FOOT
        mean = np.mean(Y,axis=0)
        upper = mean+2*std_devs
        lower = mean-2*std_devs
        mean_l,  = ax1.plot(meters,  mean,lw=lw,label="Simulated Mean")
        upper_l, = ax1.plot(meters, upper,lw=lw,label="Upper Bound")
        lower_l, = ax1.plot(meters, lower,lw=lw,label="Lower Bound")
        ax1.fill_between(x=meters,y1=upper,y2=lower,alpha=0.3)
        ax1.legend()
        plt.savefig("{:s}/variation_{:s}.png".format(out_folder,name.lower()))

    #CC vs environment factors plots
    illum_cc_poly = np.poly1d(np.polyfit(illum,p_cc,1))
    degF_cc_poly = np.poly1d(np.polyfit(degF,p_cc,1))
    RH_cc_poly = np.poly1d(np.polyfit(RH,p_cc,1))

    fig,ax1 = plt.subplots()
    ax1.scatter(illum,p_cc,marker='.',lw=0.05,color="#EDB120")
    ax1.plot(illum, illum_cc_poly(illum),"r--")
    ax1.set_title("Illuminance & CC Trend")
    ax1.set_xlabel("Illuminance [log10 Lux]")
    ax1.set_ylabel("Prediction CC")
    plt.tight_layout()

    plt.savefig("{0}/CC_trend_illuminance.png".format(out_folder))

    fig,ax1 = plt.subplots()
    ax1.scatter(degF,p_cc,marker='.',lw=0.05,color="#A2142F")
    ax1.plot(degF, degF_cc_poly(degF),"r--")
    ax1.set_title("Temperature & CC Trend")
    ax1.set_xlabel("Temperature [degrees F]")
    ax1.set_ylabel("Prediction CC")
    plt.tight_layout()
    plt.savefig("{0}/CC_trend_temperature.png".format(out_folder))

    fig,ax1 = plt.subplots()
    ax1.scatter(RH,p_cc,marker='.',lw=0.05,color="#0072BD")
    ax1.plot(RH, RH_cc_poly(RH),'r--')
    ax1.set_title("Humidity & CC Trend")
    ax1.set_xlabel("Relative Humidity [%]")
    ax1.set_ylabel("Prediction CC")
    plt.tight_layout()
    plt.savefig("{0}/CC_trend_humidity.png".format(out_folder))


    #error vs environment factors plots
    illum_mse_poly = np.poly1d(np.polyfit(illum,p_mse,1))
    degF_mse_poly = np.poly1d(np.polyfit(degF,p_mse,1))
    RH_mse_poly = np.poly1d(np.polyfit(RH,p_mse,1))

    fig,ax1 = plt.subplots()
    ax1.scatter(illum,p_mse,marker='.',lw=0.05,color="#EDB120")
    ax1.plot(illum, illum_mse_poly(illum),"r--")
    ax1.set_title("Illuminance & MSE Trend")
    ax1.set_xlabel("Illuminance [log10 Lux]")
    ax1.set_ylabel("Prediction MSE")
    plt.tight_layout()
    plt.savefig("{0}/error_trend_illuminance.png".format(out_folder))

    fig,ax1 = plt.subplots()
    ax1.scatter(degF,p_mse,marker='.',lw=0.05,color="#A2142F")
    ax1.plot(degF, degF_mse_poly(degF),"r--")
    ax1.set_title("Temperature & MSE Trend")
    ax1.set_xlabel("Temperature [degrees F]")
    ax1.set_ylabel("Prediction MSE")
    plt.tight_layout()
    plt.savefig("{0}/error_trend_temperature.png".format(out_folder))

    fig,ax1 = plt.subplots()
    ax1.scatter(RH,p_mse,marker='.',lw=0.05,color="#0072BD")
    ax1.plot(RH, RH_mse_poly(RH),'r--')
    ax1.set_title("Humidity & MSE Trend")
    ax1.set_xlabel("Relative Humidity [%]")
    ax1.set_ylabel("Prediction MSE")
    plt.tight_layout()
    plt.savefig("{0}/error_trend_humidity.png".format(out_folder))
        


    #show all plots
    if SHOW_PLOTS:
        plt.show()

def load_model(path = None):
    if path is None:
        path = askopenfilename() #open OS GUI to locate a saved model dictionary
    if path == '':
        print("ERROR in load_model(): Empty path to state dictionary received.")
        return
    network = Net()
    network.load_state_dict(torch.load(path))
    network.double()
    return network
    
if __name__ == '__main__':
    n = train()
    test(n)