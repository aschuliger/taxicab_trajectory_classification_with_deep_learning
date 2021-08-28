import pandas as pd
import numpy as np
import glob
import pickle
import matplotlib.pyplot as plt
import math
from scipy.stats import mode

#generate trajectories from CSV file 
def aggr(data):
    traj_raw = data.values[:,1:]
    traj = np.array(sorted(traj_raw,key = lambda d:d[2]))
    label = data.iloc[0][0]
    return [traj,label]

# Trying putting all of the data together into one pandas.dataframe and then work on it from there
#pre_proc = Preprocess("2016_07_01.csv")

max_longitude = 0
min_longitude = 100000
max_latitude = 0
min_latitude = 100000

filename = "Hello"

def find_boundaries(processed_data, max_longitude, min_longitude, max_latitude, min_latitude):
    for traj in processed_data:
        temp_long = max(traj[0][:,0])
        if temp_long < 114.8:
            max_longitude = max([max_longitude, max(traj[0][:,0])])
        temp_min_long = min(traj[0][:,0])
        if temp_min_long > 110:
            min_longitude = min([min_longitude, min(traj[0][:,0])])
        temp_lat = max(traj[0][:,1])
        if temp_lat < 22.9:
            max_latitude = max([max_latitude, max(traj[0][:,1])])
        temp_lat = min(traj[0][:,1])
        if temp_lat > 22.4:
            min_latitude = min([min_latitude, min(traj[0][:,1])])
        #print(min_latitude)

    return (max_longitude, min_longitude, max_latitude, min_latitude)

path = 'all_data'
filenames = glob.glob(path + "/*.csv")
for filename in filenames:
    data = pd.read_csv(filename).groupby('plate').apply(aggr)
    max_longitude, min_longitude, max_latitude, min_latitude = find_boundaries(data, max_longitude, min_longitude, max_latitude, min_latitude)
    #print(min_latitude)


longitude_distance = max_longitude - min_longitude
latitude_distance = max_latitude - min_latitude

print("Maximum Longitude:")
print(max_longitude)
print("Minimum Longitude:")
print(min_longitude)
print("Maximum Latitude:")
print(max_latitude)
print("Minimum Latitude:")
print(min_latitude)
print("Longitude Distance:")
print(longitude_distance)
print("Latitude Distance:")
print(latitude_distance)