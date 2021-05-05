#envi_test.py
#used to examine results more closely

#import torch
#import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import lines
import random
import numpy as np
from scipy.interpolate import interp1d #for interpolation
import scipy.integrate #for shaded area calculation
import datetime as dt
from scipy.stats import gaussian_kde #for kernel estimation
from mpl_toolkits.mplot3d import Axes3D  # for 3d projection plots

#======== PLOTTING CONFIG +=========
FONT_SIZE = 11
SMALL_FIG_FONT_SIZE = 16
plt.rcParams.update({'font.size': FONT_SIZE})

# ======= CONSTANTS ==========
TRAIN_RATIO = 0.75
TEST_RATIO = 1-TRAIN_RATIO
BIAS = True #should there be a linear bias component (i.e. an extra value of 1 in input)
SHUFFLE = False #randomly select training data, CURRENTLY OBSOLETE
SHOW_PLOTS = False
INTERP_DATA = False #interpolates waveforms into a spline before solving
INTERP_VISUAL = False #interpolates waveforms into a spline AFTER solving/simulating (if interp_data=True, this is ignored)
NORMALIZE=True

COMBINED_PLOT = True
PAST_PLOT = False
REFLECTION_PLOT = True

N_simulations = 10000 #number of waveforms to generate for each factor (varied factor sampled uniformly at this many points across measured range)


PAST_MEASUREMENTS = 15

FEET_PER_SAMPLE = 3.63716
METERS_PER_FOOT = 0.3048
WF_SIZE = 92
INTERP_SIZE = 1000
# ======= LOAD DATA ==========
in_path = "combined_data.csv"
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
#env_data[:,degC_index] = degC = raw_data[:,3] #temperature; left in farenheit
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
N_winter = 14595
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

p_full = x_full @ M
p_cc = np.zeros(N_full)
for i in range(N_full):
    p_cc[i] = np.corrcoef(y_full[i], p_full[i])[0,1]

p_mse = np.mean((p_full-y_full)**2,axis=1)

#zoomed CC plot
#plt.figure()
sample_gap = 2000 #gap between summer and winter, in samples
xtick_gap = 4*24*3600 #in seconds
cc_x = np.arange(0,N_env)
cc_x[N_winter:] += sample_gap
time_xticks = []
time_xticklabels = []
last_tick = times[0]
for i,t in enumerate(times):
    if t-last_tick > xtick_gap:
        last_tick = t
        time_xticks.append(cc_x[i])
        date_str = str(dt.datetime.fromtimestamp(t,dt.timezone.utc).date())
        time_xticklabels.append(date_str)
lw=1
fig,ax1 = plt.subplots()
ax1.set_xticks(time_xticks)
ax1.set_xticklabels(time_xticklabels, rotation=45, ha='right', rotation_mode="anchor")
#plot training: split training set into multiple parts with borders at index discontinuities and the season border (if present). this prevents lines from being drawn across long gaps
train_segments = np.split(train_indices,np.where(np.logical_or(np.diff(train_indices)!=1, train_indices[1:]==N_winter))[0]+1) #index train_indices at [1:] to match size with diff results. add 1 to offset this movement
for seg in train_segments:
    lt1, = ax1.plot(cc_x[seg],p_cc[seg], label="Training CC",color="C0",lw=lw)
#plot testing: split as training set is above
test_segments = np.split(test_indices,np.where(np.logical_or(np.diff(test_indices)!=1, test_indices[1:]==N_winter))[0]+1)
for seg in test_segments:
    lt2, = ax1.plot(cc_x[seg], p_cc[seg], label="Testing CC",color="C1",lw=lw)
ax1.set_ylabel("CC")
ax1.set_xlabel("Date")
ax2 = ax1.twinx()
#plot illuminance
#li, = ax2.plot(cc_x, illum,label="Illuminance",alpha=0.3, lw=1,color="C2")
li, = ax2.plot(cc_x[:N_winter], illum[:N_winter],label="Illuminance",alpha=0.3, lw=1,color="C2")
li, = ax2.plot(cc_x[N_winter:], illum[N_winter:],label="Illuminance",alpha=0.3, lw=1,color="C2")
#ax2.fill_between(range(N_full),illum,np.ones(N_full)*np.min(illum), alpha=0.2, color="C2")
ax2.fill_between(cc_x[:N_winter],illum[:N_winter],np.ones(N_winter)*np.min(illum), alpha=0.2, color="C2")
ax2.fill_between(cc_x[N_winter:],illum[N_winter:],np.ones(N_summer)*np.min(illum), alpha=0.2, color="C2")
ax2.set_ylabel("Log10 Illuminance (log Lux)")
plt.legend((lt1,lt2,li),("Training CC","Testing CC","Illuminance"),loc="lower left")
#plt.legend((lt1,lt2,li),("Training CC","Testing CC","Illuminance"),loc=(1,0))
plt.title("Linear Model Prediction CC\nTrain mean CC: {:.4f}\n Test mean CC: {:.4f}".format(np.mean(p_cc[train_indices]), np.mean(p_cc[test_indices])))
plt.tight_layout()
plt.savefig("{:s}/{:s}_cc_full_zoomed".format(out_folder,desc_str)) #zoomed on plotted region
ax1.set_ylim((0,1))
plt.savefig("{:s}/{:s}_cc_full".format(out_folder,desc_str)) #full 0-1 CC range


#violin plots showing environment distribution
plt.rcParams.update({'font.size': SMALL_FIG_FONT_SIZE})
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
plt.rcParams.update({'font.size': FONT_SIZE})


#temperature/illuminance plot
plt.figure()
plt.title("Temperature and Illuminance in Measured Data")
plt.scatter(degC,illum,marker='.',lw=0.2)
plt.xlabel("Temperature [Degrees C]")
plt.ylabel("Illuminance [log10 Lux]")
plt.savefig("{:s}/temp_illum_comparison".format(out_folder))

#=========== plotting variation from each factor ====================

#find mode daytime illuminance
#illum_counts, illum_centers = np.histogram(illum)
#illum_mode_bin_i = np.argmax(illum_counts)
#illum_mode = illum_centers[illum_mode_bin_i]
illum_x = np.linspace(np.min(illum),np.max(illum),N_simulations);
illum_kernel = gaussian_kde(illum)
illum_mode = illum_x[np.argmax(illum_kernel.pdf(illum_x))]

#find mode temperature
#degC_counts, degC_centers = np.histogram(degC)
#degC_mode_bin_i = np.argmax(degC_counts)
#degC_mode = degC_centers[degC_mode_bin_i]
degC_x = np.linspace(np.min(degC),np.max(degC),N_simulations);
degC_kernel = gaussian_kde(degC)
degC_mode = degC_x[np.argmax(degC_kernel.pdf(degC_x))]

#find mode humidity
#RH_counts, RH_centers = np.histogram(RH)
#RH_mode_bin_i = np.argmax(RH_counts)
#RH_mode = RH_centers[RH_mode_bin_i]
RH_x = np.linspace(np.min(RH),np.max(RH),N_simulations);
RH_kernel = gaussian_kde(RH)
RH_mode = RH_x[np.argmax(RH_kernel.pdf(RH_x))]

#Illuminance: fix temperature and humidity
illum_I = np.ones((N_simulations,ENV_SIZE))
illum_I[:,illum_index] = illum_x
illum_I[:,degC_index] = degC_mode
illum_I[:,RH_index] = RH_mode
illum_Y = illum_I@M

degC_I = np.ones((N_simulations,ENV_SIZE))
degC_I[:,illum_index] = illum_mode
degC_I[:,degC_index] = degC_x
degC_I[:,RH_index] = RH_mode
degC_Y = degC_I@M

RH_I = np.ones((N_simulations,ENV_SIZE))
RH_I[:,illum_index] = illum_mode
RH_I[:,degC_index] = degC_mode
RH_I[:,RH_index] = RH_x
RH_Y = RH_I @ M

N_COMBINED_SIMULATIONS = 100 #we get a number of pints equal to this number CUBED
combined_space = np.ones((N_COMBINED_SIMULATIONS,N_COMBINED_SIMULATIONS,N_COMBINED_SIMULATIONS,ENV_SIZE)) #make it 4d for now, flatten it after population
for i,il in enumerate(np.linspace(np.min(illum),np.max(illum),N_COMBINED_SIMULATIONS)):
    for j,dc in enumerate(np.linspace(np.min(degC),np.max(degC),N_COMBINED_SIMULATIONS)):
        for k,rh in enumerate(np.linspace(np.min(RH),np.max(RH),N_COMBINED_SIMULATIONS)):
            combined_space[i,j,k][illum_index] = il
            combined_space[i,j,k][degC_index] = dc
            combined_space[i,j,k][RH_index] = rh
            #combined_I[i,j,k][bias_index] = 1 #done implicitly by using ones() to initialize the array
combined_I = combined_space.reshape((-1,ENV_SIZE))
combined_Y = combined_I @ M
combined_N = len(combined_I) #should be N_COMBINED_SIMULATIONS**3
assert(combined_N == N_COMBINED_SIMULATIONS**3),"Assertion failed: combined N not what expected."

if INTERP_VISUAL and not INTERP_DATA:
    x = np.arange(WF_SIZE)
    xx = np.linspace(0,WF_SIZE-1,INTERP_SIZE)
    illum_Y = np.array([interp1d(x,wf,kind='cubic')(xx) for wf in illum_Y])
    degC_Y  = np.array([interp1d(x,wf,kind='cubic')(xx) for wf in degC_Y])
    RH_Y    = np.array([interp1d(x,wf,kind='cubic')(xx) for wf in RH_Y])
    combined_Y    = np.array([interp1d(x,wf,kind='cubic')(xx) for wf in combined_Y])

plt.rcParams.update({'font.size': SMALL_FIG_FONT_SIZE})
#each plot will share the same code. extra initial investment, but reduces headaches later.
mode_strings = ("Illuminance: 10^{:.2f} Lux".format(illum_mode), "Temperature: {:.2f} C".format(degC_mode), "Humidity: {:.2f} %".format(RH_mode))
lw = 1
for i,(name,Y) in enumerate(zip(("Illuminance","Temperature","Humidity","All"),(illum_Y,degC_Y,RH_Y,combined_Y))):
    if i!=3 or COMBINED_PLOT:
        used_m_strings = [m for j,m in enumerate(mode_strings) if i!=j and i!=3]
        fig,ax1 = plt.subplots(figsize=(6.4/0.63,4.8))
        ax1.set_xlabel("Distance (meters)")
        ax1.set_ylabel("Normalized SSTDR Magnitude")
        std_devs = np.std(Y,axis=0)
        if INTERP_DATA or INTERP_VISUAL:
            meters = np.arange(INTERP_SIZE)*FEET_PER_SAMPLE*WF_SIZE/INTERP_SIZE*METERS_PER_FOOT
            meters -= meters[int(np.argmax(wfs[0,:])/WF_SIZE*INTERP_SIZE)]
        else:
            meters = np.arange(WF_SIZE)*FEET_PER_SAMPLE*METERS_PER_FOOT
            meters -= meters[np.argmax(wfs[0,:])]
        mean = np.mean(Y,axis=0)
        upper = mean+2*std_devs
        lower = mean-2*std_devs
        area = scipy.integrate.simps(abs(upper-lower)) #TODO make this use meters as an X axis, not samples
        upper_l, = ax1.plot(meters, upper,lw=lw,label="Upper Bound",color="C1")
        mean_l,  = ax1.plot(meters,  mean,lw=lw,label="Simulated Mean",color="C0")
        lower_l, = ax1.plot(meters, lower,lw=lw,label="Lower Bound",color="C2")
        ax1.fill_between(x=meters,y1=upper,y2=lower,alpha=0.3)
        ax1.legend(loc='upper right')
        ax1.set_ylim((-1,1))
        #if i!=3: ax1.set_title("Variation Due to {:s}\n{:s}\n{:s}\nShaded Area: {:.2f}".format(name,used_m_strings[0],used_m_strings[1], area))
        #else: ax1.set_title("Variation Due to {:s}\n\n\nShaded Area: {:.2f}".format(name, area))
        ax1.set_title("Variation Due to {:s}; Shaded Area = {:.2f}".format(name, area))
        ax2 = ax1.twinx()
        ax2.plot(meters,std_devs,lw=lw, linestyle=':',color='purple')
        ax2.set_ylabel("Std. Dev")
        ax2.set_ylim((0,0.5))
        ax2.set_yticks([0.04,0.1],minor=False)
        ax2.set_yticks(np.arange(0,0.15,0.02),minor=True)
        ax2.yaxis.set_label_coords(1.11, 0.15)
        ax1.set_xlim((np.min(meters),np.max(meters)))
        ax2.set_xlim((np.min(meters),np.max(meters)))
        #ax2.minorticks_on()
        ax2.grid(True, which='both',axis='y')
        plt.tight_layout()
        plt.savefig("{:s}/variation_{:s}.png".format(out_folder,name.lower()))
plt.rcParams.update({'font.size': FONT_SIZE})


#CC vs environment factors plots
illum_cc_poly = np.poly1d(np.polyfit(illum,p_cc,1))
degC_cc_poly = np.poly1d(np.polyfit(degC,p_cc,1))
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
ax1.scatter(degC,p_cc,marker='.',lw=0.05,color="#A2142F")
ax1.plot(degC, degC_cc_poly(degC),"r--")
ax1.set_title("Temperature & CC Trend")
ax1.set_xlabel("Temperature [degrees C]")
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
degC_mse_poly = np.poly1d(np.polyfit(degC,p_mse,1))
RH_mse_poly = np.poly1d(np.polyfit(RH,p_mse,1))
ILLUM_COLOR = "#EDB120"
DEGC_COLOR = "#A2142F"
RH_COLOR = "#0072BD"

fig,ax1 = plt.subplots()
ax1.scatter(illum,p_mse,marker='.',lw=0.05,color=ILLUM_COLOR)
ax1.plot(illum, illum_mse_poly(illum),"r--")
ax1.set_title("Illuminance & MSE Trend")
ax1.set_xlabel("Illuminance [log10 Lux]")
ax1.set_ylabel("Prediction MSE")
plt.tight_layout()
plt.savefig("{0}/error_trend_illuminance.png".format(out_folder))

fig,ax1 = plt.subplots()
ax1.scatter(degC,p_mse,marker='.',lw=0.05,color=DEGC_COLOR)
ax1.plot(degC, degC_mse_poly(degC),"r--")
ax1.set_title("Temperature & MSE Trend")
ax1.set_xlabel("Temperature [degrees C]")
ax1.set_ylabel("Prediction MSE")
plt.tight_layout()
plt.savefig("{0}/error_trend_temperature.png".format(out_folder))

fig,ax1 = plt.subplots()
ax1.scatter(RH,p_mse,marker='.',lw=0.05,color=RH_COLOR)
ax1.plot(RH, RH_mse_poly(RH),'r--')
ax1.set_title("Humidity & MSE Trend")
ax1.set_xlabel("Relative Humidity [%]")
ax1.set_ylabel("Prediction MSE")
plt.tight_layout()
plt.savefig("{0}/error_trend_humidity.png".format(out_folder))


def get_closest(array, target_value,max_dist=np.inf): #returns index with value closest to target_value, within max_dist (if provided)
    dists = abs(array-target_value)
    li = np.argmin(dists)
    if dists[li] < max_dist:
        return li
    else:
        return None #in another language I might return -1 but that's a valid index in python

#========== fault reflection comparison plot =============
#original idea: tiered lines
if REFLECTION_PLOT:
    location_meters = 60
    location_index = get_closest(meters, location_meters) #get index of closest location
    fig,ax1 = plt.subplots()
    ax1.set_xlabel("Temperature [degrees C]")
    ax1.set_ylabel("Waveform Value")

    #illum_range = np.linspace(np.min(illum),np.max(illum),N_COMBINED_SIMULATIONS)
    #rh_range    = np.linspace(np.min(RH),np.max(RH),N_COMBINED_SIMULATIONS)

    degC_N = 500
    degC_range  = np.linspace(np.min(degC),np.max(degC),degC_N)
    illum_values = (4.5, -1.5)
    rh_values = (0, 50, 100)
    #illum_indices = [get_closest(illum_range, v) for v in illum_values]
    #rh_indices = [get_closest(rh_range, v) for v in rh_values]
    
    reflect_space = np.ones((len(illum_values), len(rh_values), degC_N, ENV_SIZE))
    reflect_space_Y = np.zeros((len(illum_values), len(rh_values), degC_N, WF_SIZE))
    for j,rh_v in enumerate(rh_values):
        for i,illum_v in enumerate(illum_values):
            reflect_space[i,j,:,0] = illum_v
            reflect_space[i,j,:,1] = degC_range
            reflect_space[i,j,:,2] = rh_v
            block = reflect_space[i,j,:] @ M
            #block_mean = np.mean(block,axis=0)
            reflect_space_Y[i,j,:,:] = block
            #Y_at_point = (block-block_mean)[:,location_index]
            Y_at_point = block[:,location_index]
            ax1.plot(degC_range, Y_at_point, lw=lw, label="{:4.1f} RH; 10^{:.1f} Lux".format(rh_v, illum_v)) #TODO color?
    
    ax1.legend(loc='upper right')
    #ax1.set_ylim((-1,1))
    #ax1.set_xlim((np.min(degC),np.max(degC)))
    ax1.set_title("Expected Values at {:.1f} Meters".format(meters[location_index]))
    """
    ax2 = ax1.twinx()
    ax2.plot(meters,std_devs,lw=lw, linestyle=':',color='purple')
    ax2.set_ylabel("Std. Dev")
    ax2.set_ylim((0,0.5))
    ax2.set_yticks([0.04,0.1],minor=False)
    ax2.set_yticks(np.arange(0,0.15,0.02),minor=True)
    ax2.yaxis.set_label_coords(1.11, 0.15)
    ax2.set_xlim((np.min(meters),np.max(meters)))
    """
    #ax2.minorticks_on()
    #ax2.grid(True, which='both',axis='y')
    plt.tight_layout()
    plt.savefig("{:s}/variation_reflection.png".format(out_folder))

#new idea: M slope
if REFLECTION_PLOT:
    #locations = (40, 65, 70)
    #location_indices = [get_closest(meters, l) for l in locations] #get index of closest location
    #M_list = [M[:,i] for i in location_indices]
    
    M_names = ["Median Pre-panel", "Median Post-Panel", "All Locations"]
    range_points = (get_closest(meters, 0), get_closest(meters, 60), len(meters))
    M_list = []
    M_list.append(np.median(np.abs(M[:,range_points[0]:range_points[1]]),axis=1)) #pre-panel slope mean
    M_list.append(np.median(np.abs(M[:,range_points[1]:range_points[2]]),axis=1)) #post-panel slope mean
    M_list.append(np.max(M, axis=1)) #highest slope for any point for each factor
    
    #PAPER TODO
    #[x] means and max is fine, consider adding numbers to fault labels, add labels for each colored line (make sure in order)
    #[x] make plot taller?
    #[x] make figure 4 span entire column
    #[ ] other plot: number baselines required vs each range; required to catch every tkind of fault
    #[x] citations: remove ISBNs, DOI/URLS
    #[x] use time instead of samples in figure 3
    #[ ] we want to automate environment data collection... consider long usb cable to connect the sensor to the cupola computer
    
    #[ ] Fix variation reduction: splits into 2*N portions rather than N+1 portions
    
    reflection_N = 500
    samples = np.linspace(0,100,reflection_N)
    
    illum_range = np.linspace(0,np.max(illum)-np.min(illum),reflection_N)
    degC_range  = np.linspace(0,np.max(degC)-np.min(degC),reflection_N)
    rh_range    = np.linspace(0,np.max(RH)-np.min(RH),reflection_N)
    
    fig,axes = plt.subplots(3,1,sharey=True, constrained_layout=True)
    fig.set_figheight(1.5*fig.get_figheight())
    #outline_ax = fig.add_subplot(111,frameon=False)
    #outline_ax.set_ylabel("Expected Variation")
    #outline_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    #fig.subplots_adjust(hspace=0,wspace=0)
    axes[0].set_xlabel("Maximum Illuminance Range [Δ log10 Lux]")
    axes[1].set_xlabel("Maximum Temperature Range [Δ °C]")
    axes[2].set_xlabel("Maximum Humidity Range [Δ %]")
    
    axes[0].set_xlim((0,illum_range[-1]))
    axes[1].set_xlim((0,degC_range[-1]))
    axes[2].set_xlim((0,rh_range[-1]))
    
    axes[0].set_yticks((0,0.2,0.4))
    axes[1].set_yticks((0,0.2,0.4))
    axes[2].set_yticks((0,0.2,0.4))
    
    axes[1].set_ylabel("SSTDR Variation")#don't make this too long or the subplots get squished
    handles = [None]*3
    for i,m in enumerate(M_list):
        #plot line of slope M[n] starting at 0. each n represents a particular factor.
        #axes[0].plot(illum_range, illum_range*abs(m[0]), lw=2, label="Illum Range @ "+str(location_index))
        handles[i], = axes[0].plot(illum_range, illum_range*abs(m[0]), lw=2)
        #axes[1].plot(degC_range, degC_range*abs(m[1]), lw=2, label="Temperature Range @ "+str(location_index))
        axes[1].plot(degC_range, degC_range*abs(m[1]), lw=2)
        #axes[2].plot(rh_range, rh_range*abs(m[2]), lw=2, label="Humidity Range @ "+str(location_index))
        axes[2].plot(rh_range, rh_range*abs(m[2]), lw=2)
    handles.reverse()
    M_names.reverse()
    axes[0].legend(handles, M_names, loc='upper left')
    #axes[0].set_ylim((0,1))
    #axes[1].set_ylim((0,1))
    #axes[2].set_ylim((0,1))
    #ax1.set_xlim((np.min(degC),np.max(degC)))
    axes[0].set_title("Variation and Fault Magnitude")
    #TODO plot fault reflections
    fault_names = ("Pre-panel Open", "Broken Cell", "Arc Fault", "Ground Fault", "Post-panel Open")
    open_norm = 0.32512 #the magnitude of an open fault reflection in OUR panel setup
    fault_magnitudes = np.array((1.0, 0.1, 0.3, 0.15, 0.147/open_norm)) #the last one here was measured on our panels; divide by open_norm to cancel out the normalization later
    norm_faults = fault_magnitudes*open_norm
    fault_name_dict = dict(zip(norm_faults, fault_names))
    norm_faults.sort()
    sorted_names = [fault_name_dict[f] for f in norm_faults]
    
    text_x_padding = 0.10
    text_y_padding = 0.05
    for i in range(len(axes)):
        twin = axes[i].twinx()
        twin.set_zorder(axes[i].get_zorder() -1)
        axes[i].patch.set_visible(False)
        twin.set_yticks(np.array(fault_magnitudes)*open_norm)
        twin.set_ylim(axes[i].get_ylim())
        twin.set_yticklabels([])
        twin.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        twin.grid(True, which='both',axis='y')
        xmax = axes[i].get_xlim()[1]
        ymax = axes[i].get_ylim()[1]
        texts = []
        
        #plot label text
        for f,n in zip(norm_faults, sorted_names):
            t = "{:.2f}: {:s}".format(f,n)
            if f < twin.get_ylim()[1]:
                texts.append(axes[i].text(xmax+text_x_padding*xmax,f,t,va='center',ha='left')) #normalize padding
        #arrange label text
        r = fig.canvas.get_renderer()
        text_height = texts[0].get_window_extent(renderer=r).inverse_transformed(axes[i].transData).height #height of text in plot units, not pixels
        for j in range(len(texts)):
            if j > 0:
                #move text to not overlap with lower texts
                y = max(texts[j].get_position()[1], texts[j-1].get_position()[1]+text_height+text_y_padding*ymax)
                texts[j].set_position((texts[j].get_position()[0], y))
            else:
                y = texts[j].get_position()[1]
            #draw line from ytick to text
            line = lines.Line2D((xmax,xmax+text_x_padding*xmax), (norm_faults[j],y), lw=0.5, color='gray', alpha=1)
            line.set_clip_on(False)
            twin.add_line(line)

    #plt.tight_layout()
    plt.savefig("{:s}/variation_comparison.png".format(out_folder))


#======== baselines required plot ============
#fig,axes = plt.subplots(3,1,sharey=True, constrained_layout=True)
fig,axes = plt.subplots(3,1,sharey=True)
fig.set_figheight(1.5*fig.get_figheight())

axes[0].set_xlabel("Maximum Illuminance Range [Δ log10 Lux]")
axes[1].set_xlabel("Maximum Temperature Range [Δ °C]")
axes[2].set_xlabel("Maximum Humidity Range [Δ %]")

for ax in axes:
    ax.set_yticks([1,3,5,7,9],minor=False)
    ax.set_yticks([3,5,7],minor=True)

axes[0].set_xlim((0,illum_range[-1]))
axes[1].set_xlim((0,degC_range[-1]))
axes[2].set_xlim((0,rh_range[-1]))

outline_ax = fig.add_subplot(111,frameon=False)
outline_ax.set_ylabel("Minimum Baselines Required s.t. Variation ≤ Faults")
outline_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

#fig.subplots_adjust(hspace=0,wspace=0)
handles = [None]*3
for i,m in enumerate(M_list):
    #plot line of slope M[n] starting at 0. each n represents a particular factor.
    #old formula: n = ceil(range / (smallest_fault_variation / slope)) - 1
    #assumed N baselines split the range into N+1 pieces,
    #producing variation equal to the full range variation / (N+1)
    #incorrectly allocates left/right sides of baselines 'territory' to overlap.
    #num_illum = np.ceil((illum_range / (norm_faults[0] / m[0])) - 1)
    #num_degC  = np.ceil(( degC_range / (norm_faults[0] / m[1])) - 1)
    #num_rh    = np.ceil((   rh_range / (norm_faults[0] / m[2])) - 1)
    
    #new formula: n = ceil(slope*range/2/smallest_fault)
    #assumes N baselines split the variation into N regions, with each baseline at the center,
    #producing variation equal to the full range variation / N / 2. algebra steps below
    #full_var = slope*range
    #partial_var = slope*partial_range = slope*range/N/2
    #smallest_fault >= slope*range/N/2
    #N >= slope*range/2/smallest_fault
    
    num_illum = np.ceil(m[0]*illum_range/2/norm_faults[0])
    num_degC  = np.ceil(m[1]* degC_range/2/norm_faults[0])
    num_rh    = np.ceil(m[2]*   rh_range/2/norm_faults[0])
    
    num_illum[num_illum < 1] = 1
    num_degC [num_degC  < 1] = 1
    num_rh   [num_rh    < 1] = 1
    handles[i], = axes[0].plot(illum_range, num_illum, lw=2)
    axes[1].plot(degC_range, num_degC, lw=2)
    axes[2].plot(rh_range, num_rh, lw=2)
handles.reverse()
axes[0].legend(handles,M_names, loc='upper left')

axes[0].grid(True, which='both',axis='y')
axes[1].grid(True, which='both',axis='y')
axes[2].grid(True, which='both',axis='y')

plt.tight_layout()
plt.savefig("{:s}/baseline_requirements.png".format(out_folder))


#========== PAST-INFORMED LINEAR MODEL =============
if PAST_PLOT:    
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
ax.set_ylabel("Temperature (C)")
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
plt.title("Full Waveform CC")
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
plt.title("Pruned Region CC")
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

