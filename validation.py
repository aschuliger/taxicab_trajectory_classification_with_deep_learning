import pandas as pd
import numpy as np
import glob
import pickle
import matplotlib.pyplot as plt
import math
from scipy.stats import mode
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, TensorDataset

device = torch.device("cuda")

class CustomTensorDataset(Dataset):
    
    def __init__(self, tensors, transform=None):
        #assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        service_traj = torch.tensor(self.tensors[0][index][1],dtype=torch.float32)
        seek_traj = torch.tensor(self.tensors[0][index][0],dtype=torch.float32)
        feat = torch.tensor(self.tensors[0][index][2],dtype=torch.float32)

        y = self.tensors[1][index]

        return (seek_traj, service_traj, feat), y

    def __len__(self):
        return len(self.tensors[0])

num_sub_traj_seek = 70
#define your model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.rnn_seek = nn.RNN(4,2,batch_first=True)
        self.rnn_service = nn.RNN(4,2,batch_first=True)
        #self.fc1 = nn.Linear(5,10)
        #self.fc2 = nn.Linear(2*num_sub_traj_seek+2*num_sub_traj_seek+10,5)
        self.fc2 = nn.Linear(2*num_sub_traj_seek+2*num_sub_traj_seek+5,5)

    def forward(self, seek, service, features):
        seek_rearranged = seek.permute(1,0,2,3)
        service_rearranged = service.permute(1,0,2,3)
        #emb_fea = self.fc1(features)

        seek_output = [self.rnn_seek(cell)[1][0] for cell in seek_rearranged]
        service_output = [self.rnn_service(cell)[1][0] for cell in service_rearranged]

        #input_tensor = torch.cat(service_output+[emb_fea],axis=1)
        input_tensor = torch.cat(service_output+[features],axis=1)
        input_tensor = torch.cat(seek_output+[input_tensor],axis=1)

        return F.softmax(self.fc2(input_tensor), dim=0)


def check_accuracy(testing, loader, model):
    if testing:
        print("Checking accuracy on test data")
    else:
        print("Checking accuracy on training data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            seek_traj = x[0].to(device)
            service_traj = x[1].to(device)
            feat = x[2].to(device)
            labels = y.to(device)

            scores = model(seek_traj, service_traj, feat)
            _, predictions = scores.max(1)
            num_correct += (predictions == labels).sum()
            num_samples += predictions.size(0)

            print(f'Label: {labels}')
            print(f'Prediction: {predictions}')

        print(f'Got {num_correct} / {num_samples} with accuracy \ {float(num_correct)/float(num_samples)*100:.2f}')
    
    model.train()

path = 'preprocessed_data'
filenames = glob.glob(path + "/*.pickle")
testing = []
test_labels = []
for filename in filenames:
    data = pickle.load( open(filename, "rb") )
    testing_data = data[0]
    test_label = data[1]
    for traj, traj_label in zip(testing_data, test_label):
        testing.append(traj)
        test_labels.append(traj_label)

testset = CustomTensorDataset(tensors=(testing, torch.tensor(test_labels,dtype=torch.long)))
testloader = torch.utils.data.DataLoader(testset, batch_size=50, shuffle=True)

model = Net().to(device)
model.load_state_dict(torch.load("final_model.pth"))

check_accuracy(True, testloader, model)