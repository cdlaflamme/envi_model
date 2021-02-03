#envi_test.py
#used to examine results more closely

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import numpy as np
from scipy.interpolate import interp1d

from mpl_toolkits.mplot3d import Axes3D  # for 3d projection plots

# ======= CONSTANTS ==========
TRAIN_RATIO = 0.75
TEST_RATIO = 1-TRAIN_RATIO
BIAS = True #should there be a linear bias component (i.e. an extra value of 1 in input)
SHUFFLE = False #randomly select training data, CURRENTLY OBSOLETE
SHOW_PLOTS = False
INTERP_DATA = False #interpolates waveforms into a spline before solving
INTERP_VISUAL = True #interpolates waveforms into a spline AFTER solving/simulating (if interp_data=True, this is ignored)
NORMALIZE=True

PAST_MEASUREMENTS = 15

FEET_PER_SAMPLE = 3.63716
METERS_PER_FOOT = 0.3048
WF_SIZE = 92
INTERP_SIZE = 1000
# ======= LOAD DATA ==========
in_path = "combined_data_new.csv"
#in_path = "combined_data_ends_1_21.csv"
out_folder = "linear_plots"
if (BIAS):
    ENV_SIZE = 4
else:
    ENV_SIZE = 3
raw_data = np.genfromtxt(in_path,delimiter=',',skip_header=1)
times = raw_data[:,0]
N_env = len(times)
env_data = np.zeros((N_env,ENV_SIZE))

illum_index = 0
degC_index = 1
RH_index = 2
bias_index = 3

env_data[:,illum_index] = np.where(raw_data[:,1]==0, 0.01, raw_data[:,1]) #replace illuminance values of 0 with 0.01: lowest nonzero value ever observed; needed for log
env_data[:,illum_index] = illum = np.log10(env_data[:,0]) #log10 illuminance
#env_data[:,illum_index] = raw_data[1:,1] #illuminance
env_data[:,degC_index] = degC = (raw_data[:,3]-32)*5/9 #temperature; converted to celcius
env_data[:,RH_index] = RH = raw_data[:,4] #humidity
if BIAS:
    env_data[:,bias_index] = 1
wfs_raw = wfs = raw_data[:,5:]
if NORMALIZE:
    wf_maxes = np.max(wfs_raw,axis=1)
    wfs = np.array([wfs_raw[i]/wf_maxes[i] for i in range(N_env)])
wfs_p = wfs[:,20:]

if INTERP_DATA:
    x = np.arange(WF_SIZE)
    xx = np.linspace(0,WF_SIZE-1,INTERP_SIZE)
    wfs = np.array([interp1d(x,wf,kind='cubic')(xx) for wf in wfs])
    
#============ SPLIT DATA ==============
N_winter = 14601
N_summer = N_env-N_winter
x_full = env_data
y_full = wfs
full_indices = np.array(range(N_env))
#jan_border  = int(N_winter*TEST_RATIO)
#summer_border = int(N_winter+N_summer*TRAIN_RATIO)
#best arrangement found so far: last winter, last summer

#winter_test_segment = np.array(range(0,int(N_winter*TEST_RATIO))) #first portion of winter
#winter_test_segment = np.array(range(int(N_winter*TRAIN_RATIO),N_winter)) #last portion of winter
winter_test_segment = np.arange(1625,1625+int(N_winter*TEST_RATIO))
#summer_test_segment = np.array(range(N_winter,N_winter+int(N_summer*TEST_RATIO))) #first portion of summer
#summer_test_segment = np.array(range(int(N_winter+N_summer*TRAIN_RATIO),N_env)) #last portion of summer
summer_test_segment = np.arange(36359-int(N_summer*TEST_RATIO),36359)

#winter start: 2463 for temp, 1625 for rh
#summer end: 36359

test_indices = np.concatenate((winter_test_segment,summer_test_segment))
#test_indices = full_indices[int(N_env*TRAIN_RATIO):]
train_indices = np.setdiff1d(full_indices,test_indices,assume_unique=True)

winter_train_segment = np.setdiff1d(train_indices, range(N_winter,N_env))
summer_train_segment = np.setdiff1d(train_indices, range(0,N_winter))

#if SHUFFLE:    
#    np.random.shuffle(full_indices)
#train_indices = full_indices[jan_border:summer_border]
#test_indices = np.concatenate((full_indices[0:jan_border], full_indices[summer_border:]))
    
x_train = x_full[train_indices]
x_test = x_full[test_indices]
y_train = y_full[train_indices]
y_test = y_full[test_indices]

N_train = len(x_train)
N_test = len(x_test)
N_full = N_train + N_test
assert(N_full == len(x_full))

#========== LINEAR MODEL =============
#linear model plot
#Nx92 = Nx4 @ 4x92
#Y = XM
#X.pinv Y = M
desc_str = "linear_model_TR_"+str(int(TRAIN_RATIO*100))
M = np.linalg.pinv(x_train) @ y_train

lp_full = x_full @ M
p_cc = np.zeros(N_test+N_train)
for i in range(N_full):
    p_cc[i] = np.corrcoef(y_full[i], lp_full[i])[0,1]
#plt.figure()
#plt.plot(train_indices, p_cc[train_indices],label="Training")
#plt.plot(test_indices, p_cc[test_indices],label="Testing")
#plt.plot(env_data[:,0]/max(env_data[:,0]), alpha=0.5, label="Illuminance")
#plt.title("Linear Model Prediction CC")
#plt.ylim((0,1))
#plt.ylabel("CC")
#plt.xlabel("Sample")
#plt.legend()
#plt.tight_layout()
#plt.savefig("{:s}/{:s}_cc_full".format(out_folder,desc_str))


#zoomed
#plt.figure()
lw=1
fig,ax1 = plt.subplots()
#plot training
train_segments = np.split(train_indices,np.where(np.diff(train_indices)!=1)[0]+1)
for seg in train_segments:
    lt1, = ax1.plot(seg,p_cc[seg], label="Training CC",color="C0",lw=lw)
#plot testing
test_segments = np.split(test_indices,np.where(np.diff(test_indices)!=1)[0]+1)
for seg in test_segments:
    lt2, = ax1.plot(seg, p_cc[seg], label="Testing CC",color="C1",lw=lw)

ax1.set_ylabel("CC")
ax1.set_xlabel("Sample")
ax2 = ax1.twinx()
li, = ax2.plot(illum,label="Illuminance",alpha=0.3, lw=1,color="C2")
ax2.fill_between(range(N_full),illum,np.ones(N_full)*np.min(illum), alpha=0.2, color="C2")
ax2.set_ylabel("Log10 Illuminance (log Lux)")
plt.legend((lt1,lt2,li),("Training CC","Testing CC","Illuminance"),loc="lower left")
plt.title("Linear Model Prediction CC\nTrain mean CC: {:.4f}\n Test mean CC: {:.4f}".format(np.mean(p_cc[train_indices]), np.mean(p_cc[test_indices])))
plt.tight_layout()
plt.savefig("{:s}/{:s}_cc_full".format(out_folder,desc_str))
ax1.set_ylim((0,1))
plt.savefig("{:s}/{:s}_cc_full_zoomed".format(out_folder,desc_str))

#violin plots showing environment distribution
env_sets = (illum, degC, RH)
colors = ("#EDB120","#A2142F","#0072BD")
names = ("Illuminance","Temperature","Humidity")
labels = ("Training Set","Testing Set","All Data")
units = ("log10 Lux","Degrees C", "% Relative Humidity")
for e,c,n,u in zip(env_sets,colors,names,units):
    fig,ax = plt.subplots()
    plt.title(n+" Distribution")
    vp = plt.violinplot((e[train_indices], e[test_indices], e))
    for body in vp['bodies']:
        body.set_color(c)
    for partname in ('cbars','cmins','cmaxes','cmeans','cmedians'):
        if partname in vp.keys():
            part = vp[partname]
            part.set_edgecolor(c)
            part.set_lw(1)
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    plt.ylabel(u)
    plt.tight_layout()
    plt.savefig("{:s}/distribution_{:s}".format(out_folder,n.lower()))

#temperature/illuminance plot
plt.figure()
plt.title("Temperature and Illuminance in Measured Data")
plt.scatter(degC,illum,marker='.',lw=0.2)
plt.xlabel("Temperature [Degrees C]")
plt.ylabel("Illuminance [log10 Lux]")
plt.savefig("{:s}/temp_illum_comparison".format(out_folder))

#=========== plotting variation from each factor ====================
N_simulations = 10000 #number of waveforms to generate for each factor (varied factor sampled uniformly at this many points across measured range)

#find mode daytime illuminance
illum_counts, illum_centers = np.histogram(illum)
illum_mode_bin_i = np.argmax(illum_counts)
illum_mode = illum_centers[illum_mode_bin_i]

#find mode temperature
degC_counts, degC_centers = np.histogram(degC)
degC_mode_bin_i = np.argmax(degC_counts)
degC_mode = degC_centers[degC_mode_bin_i]

#find mode humidity
RH_counts, RH_centers = np.histogram(RH)
RH_mode_bin_i = np.argmax(RH_counts)
RH_mode = RH_centers[RH_mode_bin_i]

#Illuminance: fix temperature and humidity
illum_I = np.ones((N_simulations,ENV_SIZE))
illum_I[:,illum_index] = np.linspace(np.min(illum),np.max(illum),N_simulations);
illum_I[:,degC_index] = degC_mode
illum_I[:,RH_index] = RH_mode
illum_Y = illum_I@M

degC_I = np.ones((N_simulations,ENV_SIZE))
degC_I[:,illum_index] = illum_mode
degC_I[:,degC_index] = np.linspace(np.min(degC),np.max(degC),N_simulations);
degC_I[:,RH_index] = RH_mode
degC_Y = degC_I@M

RH_I = np.ones((N_simulations,ENV_SIZE))
RH_I[:,illum_index] = illum_mode
RH_I[:,degC_index] = degC_mode
RH_I[:,RH_index] = np.linspace(np.min(RH),np.max(RH),N_simulations);
RH_Y = RH_I @ M

if INTERP_VISUAL and not INTERP_DATA:
    x = np.arange(WF_SIZE)
    xx = np.linspace(0,WF_SIZE-1,INTERP_SIZE)
    illum_Y = np.array([interp1d(x,wf,kind='cubic')(xx) for wf in illum_Y])
    degC_Y  = np.array([interp1d(x,wf,kind='cubic')(xx) for wf in degC_Y])
    RH_Y    = np.array([interp1d(x,wf,kind='cubic')(xx) for wf in RH_Y])

#each plot will share the same code. extra initial investment, but reduces headaches later.
mode_strings = ("Illuminance: 10^{:.2f} Lux".format(illum_mode), "Temperature: {:.2f} C".format(degC_mode), "Humidity: {:.2f} %".format(RH_mode))
lw = 1
for i,(name,Y) in enumerate(zip(("Illuminance","Temperature","Humidity"),(illum_Y,degC_Y,RH_Y))):
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



#========== PAST-INFORMED LINEAR MODEL =============
#advanced linear model that uses past data
ROW_SIZE = PAST_MEASUREMENTS*ENV_SIZE
desc_str = "advanced_linear_model_NM_"+str(PAST_MEASUREMENTS)+"_TR_"+str(int(TRAIN_RATIO*100))


#this method has trouble with the dataset: the first 14,601 samples were measured 1/minute; the remainder were captured at 1/2minutes.
#for the first 14,601 samples, use every other past sample, so the same "method" can be learned for both periods of data collection
X_full = np.zeros((N_full, ROW_SIZE))
skipped_measurements = PAST_MEASUREMENTS*2
for i in range(skipped_measurements-1, N_full): #start at PAST_MEASUREMENTS*2 bc the first (this many) samples don't have enough past data
    if (i <= N_winter):
        #use every other past measurement as if we captured data once per two minutes instead of once per minute
        X_full[i] = x_full[i-PAST_MEASUREMENTS*2+1:i+1:2].flatten()
    else: #we actually captured data once per 2 minutes. no need to do anything special.
        X_full[i] = x_full[i-PAST_MEASUREMENTS+1:i+1].flatten()
#want to solve for MM that turns X in to Y:
#Y = XMM
#X.pinv Y = MM
X_train = X_full[train_indices]

MM = np.linalg.pinv(X_train) @ y_train
l_adv_predictions_full = X_full @ MM
l_adv_cc = np.zeros(N_full)
for i in range(skipped_measurements-1,N_full):
    l_adv_cc[i] = np.corrcoef(y_full[i], l_adv_predictions_full[i])[0,1]

plt.figure()
plt.plot(p_cc, lw=1, label="Without Past Data")
valid_adv_cc = l_adv_cc[skipped_measurements-1:]
plt.plot(range(skipped_measurements-1,N_full),valid_adv_cc,lw=1,label="With Past Data")
#ylims = plt.ylim()
#plt.plot([border, border], [-0.5, 1.5],"r:",lw=1,label="Training Cutoff")
#plt.ylim(ylims)
#plt.plot(env_data[:,0]/max(env_data[:,0]), alpha=0.5, label="Illuminance")
plt.title("Linear Model Prediction CC, Using {:d} Minutes of Data\nMean w/o past data:  {:.4f}\nMean with past data: {:.4f}".format(PAST_MEASUREMENTS*2,np.mean(p_cc), np.mean(valid_adv_cc)))
plt.ylabel("CC")
plt.xlabel("Sample")
plt.legend()
plt.tight_layout()
plt.savefig("{:s}/{:s}_cc_full".format(out_folder,desc_str))

"""
plt.figure()
plt.plot(l_adv_cc[0:N_train], label="Training")
plt.plot(range(N_train, N_full), l_adv_cc[N_train:N_full], label="Testing")
plt.plot(env_data[:,0]/max(env_data[:,0]), alpha=0.5, label="Illuminance")
plt.title("Linear Model Prediction CC, Using "+str(PAST_MEASUREMENTS)+"Minutes of Data")
plt.ylim((0,1))
plt.ylabel("CC")
plt.xlabel("Sample")
plt.legend()
plt.savefig("{:s}/{:s}_cc_full".format(out_folder,desc_str))
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
plt.title("Full Waveform C.C.")
plt.plot(net_cc, label="Unpruned During Training")
plt.plot(net_p_cc, label = "Pruned During Training")
ylims = plt.ylim((0.8,1))
norm_illuminance = np.array(env_data[:,0])
norm_illuminance -= min(norm_illuminance)
norm_illuminance = norm_illuminance/max(norm_illuminance)*(ylims[1]-ylims[0])+ylims[0]
plt.plot(norm_illuminance,":",label="Illuminance",alpha=0.5)
plt.legend(loc='lower left')
plt.savefig("{:s}/full_waveform_prune_comparison".format(out_folder))

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
plt.savefig("{:s}/pruned_region_prune_comparison".format(out_folder))
"""
if SHOW_PLOTS:
    plt.show()

