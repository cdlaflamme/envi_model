#GNV_data_reader.py
#loads data from GNV output; converts timestamps into epoch time to align with SSTDR measurements

#data source, with info: https://mesonet.agron.iastate.edu/request/download.phtml?network=FL_ASOS
"""
GNV columns:
    station: station; GNV airport
    valid: timestamp at which data is valid 
    tmpf: temperature, Farenheit
    dwpf: dew point, Farenheit
    relh: relative humidity
    drct: wind direction, angle (degrees) from north
    sknt: wind speed, knots
    p01i: precipitation in inches "for the period from the observation time to the time of the previous hourly precipitation reset. This varies slightly by site." (not sure exactly how this works at GNV.
    alti: altitude, inches
    mslp: sea level pressure, milibar
    vsby: visibility, miles 
    gust: wind gust in knots
    [other stuff not described here; it's sky coverage, ice accredation, and peak wind info)
"""

#imports
import datetime as dt
import numpy as np
import scipy.io #for mat file saving

#read data
gnv_path = 'GNV.csv'
gnv_data = np.genfromtxt(gnv_path,delimiter=',',skip_header=1)
env_path = 'combined_data.csv'
env_data = np.genfromtxt(env_path,delimiter=',',skip_header=1)

#read environment data (measured by us)
env_timestamps = env_data[:,0]
illum   = env_data[:,1]
degF    = env_data[:,3]
rh      = env_data[:,4]
wfs     = env_data[:,5:]
precip  = gnv_data[:,7]

#read GNV data (with rain)
raw_precip_times = np.genfromtxt(gnv_path,delimiter=',',skip_header=1,dtype=str)[:,1]
raw_precip = gnv_data[:,7]

#filter out entries without precipitation
rain_indexer = ~np.isnan(raw_precip) #note the tilde; binary negation
precip = raw_precip[rain_indexer]
precip_times = raw_precip_times[rain_indexer]

#sync timestamps
#this line requires python 3.7 or higher
precip_timestamps = [dt.datetime.timestamp(dt.datetime.fromisoformat(pt)) for pt in precip_times]

scipy.io.savemat('env_with_rain.mat', {'waveforms':wfs, 'precip':precip, 'wf_times':env_timestamps, 'precip_times':precip_timestamps})