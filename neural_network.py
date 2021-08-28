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

start_time = time.perf_counter()

# Trying putting all of the data together into one pandas.dataframe and then work on it from there
#pre_proc = Preprocess("2016_07_01.csv")

#path = 'prep_data'
path = 'prep_data'
filenames = glob.glob(path + "/*.pickle")
training = []
labels = []
for filename in filenames:
    data = pickle.load( open(filename, "rb") )
    training_data = data[0]
    label = data[1]
    for traj, traj_label in zip(training_data, label):
        training.append(traj)
        labels.append(traj_label)

#Generate some random data
#print(len(training))
'''
training =[]
labels = []
num_trip = 70
len_trip = 675
for _ in range(30):
    labels.append(np.random.randint(5))
    
    #trajs is a randomly generated sequence data
    trajs = np.random.rand(num_trip,len_trip,2)
    trajs_2 = np.random.rand(num_trip, 400, 2)
    
    #feat denotes feature vector you generated for the trajectory
    feat = np.random.rand(5)
    
    training.append([trajs,trajs_2,feat])
'''

# Concatenate all data into one DataFrame
#processed_data = pd.concat(dfs, ignore_index=True)

#processed_data = pre_proc.data.groupby('plate').apply(aggr)

num_layers = 1


# Better to cut off the sequence rather than padding it with zeros
# Try different padding methods (do it yourself)
# Keep the repeat cells in your grid training representation

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


#Create Dataset for pytorch training 

num_sub_traj_seek = 70

#define your model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.rnn_seek = nn.RNN(4,2,batch_first=True)
        self.rnn_service = nn.RNN(4,2,batch_first=True)
        #self.fc1 = nn.Linear(5,3)
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

data_length = len(training)
test_length = round(data_length / 5)

print(test_length)
print(len(training))

for i in range(5):
    print(i)

    training_set = training[(i+1)*test_length:]
    labels_set = labels[(i+1)*test_length:]
    if i == 4:
        training_set = training[0:i*test_length]
        labels_set = labels[0:i*test_length]
    elif i != 0:
        training_set = np.concatenate((training[0:i*test_length],training[(i+1)*test_length:]),axis=0)
        labels_set = np.concatenate((labels[0:i*test_length],labels[(i+1)*test_length:]),axis=0)

    trainset = CustomTensorDataset(tensors=(training_set, torch.tensor(labels_set,dtype=torch.long)))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)

    
    test_set = training[i*test_length:(i+1)*test_length]
    test_labels_set = labels[i*test_length:(i+1)*test_length]

    testset = CustomTensorDataset(tensors=(test_set, torch.tensor(test_labels_set,dtype=torch.long)))
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True)


    
    net = Net().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)#, momentum=0.9)

    #train your model

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

    #check_accuracy(trainloader, net)

    for epoch in range(7):  # loop over the dataset multiple times

        running_loss = 0.0

        for x,y in trainloader:
            seek_traj = x[0].to(device)
            service_traj = x[1].to(device)
            feat = x[2].to(device)
            labels_y = y.to(device)
            #print(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(seek_traj, service_traj, feat)
            #print(outputs)
            loss = criterion(outputs, labels_y)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        print('[%d] loss: %.3f' %
                (epoch + 1, running_loss / 2000))

    end_time = time.perf_counter()
    print(f"Training took {end_time - start_time:0.4f} seconds")

    check_accuracy(False, trainloader, net)

    print("This model was built with RNNs with an output layer of 2 and a learning rate of 0.01")

    check_accuracy(True, testloader, net)


    #save model
    model_name = "validation_model_" + str(i+1) + ".pth"
    torch.save(net.state_dict(),model_name)
