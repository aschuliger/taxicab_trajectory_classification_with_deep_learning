import sys
import pandas as pd
import numpy as np
import glob
import pickle
import matplotlib.pyplot as plt
import math
from scipy.stats import mode

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, TensorDataset

def aggr(data):
        traj_raw = data.values[:,1:]
        traj = np.array(sorted(traj_raw,key = lambda d:d[2]))
        label = data.iloc[0][0]
        return [traj,label]

def preprocess_file(filename):
    processed_data = pd.read_csv(filename).groupby('plate').apply(aggr)

    max_longitude = 114.51741
    min_longitude = 113.258
    max_latitude = 22.90
    min_latitude = 22.4704

    for traj in processed_data:
        trajs = traj[0]
        for i in range(len(trajs)):
            if trajs[i][1] > max_latitude:
                np.delete(trajs, i)
            elif trajs[i][0] > max_longitude:
                np.delete(trajs, i)
            elif trajs[i][1] < min_latitude:
                np.delete(trajs, i)
            elif trajs[i][0] < min_longitude:
                np.delete(trajs, i)

    num_sub_traj_seek = 70
    sequence_length_seek = 400
    num_sub_traj_service = 70
    sequence_length_service = 200
    num_layers = 1

    longitude_distance = max_longitude - min_longitude
    latitude_distance = max_latitude - min_latitude

    def find_cell(traj):
        x = math.floor((traj[1]-min_latitude) / (latitude_distance / 500.0))
        y = math.floor((traj[0]-min_longitude) / (longitude_distance / 500.0))
        return [x, y]

    def calculate_distance(lat1, lat2, long1, long2):
        return math.sqrt(pow(lat2 - lat1, 2) + pow(long2 - long1, 2))

    def calculate_time(date1, date2):
        hours = int(date2[11:13]) - int(date1[11:13])
        mins = int(date2[14:16]) - int(date1[14:16])
        secs = int(date2[17:19]) - int(date1[17:19])
        return hours*60*60 + mins*60 + secs
        
    def get_hour(date1):
        return int(date1[11:13])

    def calculate_speed(lat1, lat2, long1, long2, date1, date2):
        time = calculate_time(date1, date2)
        if time != 0:
            return calculate_distance(lat1, lat2, long1, long2) / time
        return 0

    grid_array = []

    training_grids_seek = []
    training_grids_service = []

    for traj in processed_data:
        trajs = traj[0]
        flag = traj[0][0][3]
        sub_traj = []
        last_sub_traj = []
        cell_traj = []
        cell_traj_seek = []
        cell_traj_service = []
        grid_array_spec = []
        first_time = traj[0][0][2]
        last_time = []
        distance = 0
        past_cell = []

        for point in trajs:
            if point[3] != flag:
                if len(sub_traj) > 2:
                    if flag == 0:
                        cell_traj_seek.append(sub_traj)
                    else:
                        cell_traj_service.append(sub_traj)
                    last_sub_traj = sub_traj
                    sub_traj = []
                elif last_sub_traj != []:
                    if flag == 0:
                        cell_traj_service.remove(last_sub_traj)
                    else:
                        cell_traj_seek.remove(last_sub_traj)
                    sub_traj = last_sub_traj
                else:
                    sub_traj = []
                flag = point[3]
            cell = find_cell(point)
            grid_array_spec.append(cell)
            #print(grid_array_spec)
            speed = 0
            if past_cell != []:
                speed = calculate_speed(past_cell[0], cell[0], past_cell[1], cell[1], point[2], last_time)
            features = [cell[0], cell[1], get_hour(point[2]), speed]
            sub_traj.append(features)
            last_time = point[2]
            past_cell = [cell[0], cell[1]]
            cell = []
        if len(sub_traj) > 2:
            if flag == 0:
                cell_traj_seek.append(sub_traj)
            else:
                cell_traj_service.append(sub_traj)
        label = traj[1]
        grid_array.append(grid_array_spec)

        training_grids_seek.append(cell_traj_seek)
        training_grids_service.append(cell_traj_service)

    training = []
    labels = []

    def find_mode_cell(traj):
        x_mode, counts = mode(np.array(grid_array[traj[1]]))
        return np.array(x_mode)[0]

    def average_distances(seek_traj, service_traj):
        seeking_distances = []
        service_distances = []

        for sub_traj in seek_traj:
            distance = 0
            last_cell = sub_traj[0]
            for cell in sub_traj:
                if cell != last_cell:
                    distance = distance + 1
                    last_cell = cell
            seeking_distances.append(distance)

        for sub_traj in service_traj:
            distance = 0
            last_cell = sub_traj[0]
            for cell in sub_traj:
                if cell != last_cell:
                    distance = distance + 1
                    last_cell = cell
            service_distances.append(distance)
        
        avg_seeking = 0
        avg_service = 0
        if len(seeking_distances) != 0:
            avg_seeking = sum(seeking_distances) / len(seeking_distances)
        if len(service_distances) != 0:
            avg_service = sum(service_distances) / len(service_distances)
        return [avg_seeking, avg_service]

    features = []

    for traj, seek_traj, service_traj in zip(processed_data, training_grids_seek, training_grids_service):
        avg_seeking, avg_service = average_distances(seek_traj, service_traj)
        longitude_cell, latitude_cell = find_mode_cell(traj)
        feature = [longitude_cell, latitude_cell, avg_seeking, avg_service, len(service_traj)]
        label = traj[1]
        features.append(feature)
        labels.append(label)

    # This goes through each sub-trajectory and pads it as necessary

    for traj in training_grids_seek:
        while len(traj) > num_sub_traj_seek:
            traj.remove(traj[len(traj)-1])
        while len(traj) < num_sub_traj_seek:
            traj.append(traj[np.random.randint(len(traj))])
        for sub_traj in traj:
            while len(sub_traj) > sequence_length_seek:
                sub_traj.remove(sub_traj[len(sub_traj)-1])
            while len(sub_traj) < sequence_length_seek:
                sub_traj.append(sub_traj[len(sub_traj)-1])

    for traj in training_grids_service:
        while len(traj) > num_sub_traj_service:
            traj.remove(traj[len(traj)-1])
        while len(traj) < num_sub_traj_service:
            traj.append([[-1,-1,0,0]])
        for sub_traj in traj:
            while len(sub_traj) > sequence_length_service:
                sub_traj.remove(sub_traj[len(sub_traj)-1])
            while len(sub_traj) < sequence_length_service:
                sub_traj.append(sub_traj[len(sub_traj)-1])

    for feature, seek_traj, service_traj in zip(features, training_grids_seek, training_grids_service):
        training.append([seek_traj, service_traj, feature])

    return [training, labels]


path = sys.argv[1]
filenames = glob.glob(path + "/*.csv")
for filename in filenames:
    new_name = "preprocessed_data/prep_data" + filename[:-4][-10:] + ".pickle"
    print(new_name)
    data = preprocess_file(filename)

    with open(new_name, 'wb') as f:
        pickle.dump(data, f)
