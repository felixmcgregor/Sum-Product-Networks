import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import random
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
from NumpyDataset import NumpyDataset
from TPN import TPN

from Node import Node
from SumNode import SumNode
from ProductNode import ProductNode
from Node import LeafNode
from SPN import SPN

# ignore warnings
import warnings
warnings.filterwarnings("ignore")


# XOR data
##########################################################
y = np.array([0, 1, 1, 0])


x1 = np.array([0.0, 0.0, 0.0])
x2 = np.array([0.0, 1.0, 0.0])
x3 = np.array([1.0, 0.0, 0.0])
x4 = np.array([1.0, 1.0, 0.0])

x5 = np.array([0.0, 0.0, 1.0])
x6 = np.array([0.0, 1.0, 1.0])
x7 = np.array([1.0, 0.0, 1.0])
x8 = np.array([1.0, 1.0, 1.0])
xordata = np.array([x1, x2, x3, x4, x5, x6, x7, x8])
##########################################################
# Define TPN
##########################################################
# define hyperparameters
batch_size = 800

# define dataloader
# torchdata = NumpyDataset(xordata, y)
torchdata = NumpyDataset(xordata)
train_loader = DataLoader(torchdata, batch_size=batch_size, shuffle=True, num_workers=4)

# define neural network
tpn = TPN()
tpn.print_weights()
tpn = tpn.cuda()

# define loss function
# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(tpn.weights, lr=0.01)
#optimizer = torch.optim.Adam(tpn.weights, lr=0.01)

print("Prob x1", torch.exp(tpn(x1)).cpu().detach().numpy())
print("Prob x2", torch.exp(tpn(x2)).cpu().detach().numpy())
print("Prob x3", torch.exp(tpn(x3)).cpu().detach().numpy())
print("Prob x4", torch.exp(tpn(x4)).cpu().detach().numpy())
print("Prob x5", torch.exp(tpn(x5)).cpu().detach().numpy())
print("Prob x6", torch.exp(tpn(x6)).cpu().detach().numpy())
print("Prob x7", torch.exp(tpn(x7)).cpu().detach().numpy())
print("Prob x8", torch.exp(tpn(x8)).cpu().detach().numpy())
##########################################################
# Train
##########################################################

print("\nTraining...\n")
num_epochs = 1000
for epoch in range(num_epochs):

    for i, instances in enumerate(train_loader):
        instances = Variable(instances[0])
        instances = instances.float()

        if torch.cuda.is_available():
            # if False:
            instances = instances.cuda()
            # labels = labels.cuda()

        optimizer.zero_grad()
        outputs = -tpn(instances)
        outputs.backward(retain_graph=True)

        zero = torch.tensor([0]).float()  # .view(batch_size, 1)
        # zero = torch.tensor(float('-inf'))
        #loss = criterion(outputs, zero)
        #print("Loss", loss, "Output", outputs)
        #print("Input", instances.data, "Out", torch.exp(outputs.data[0]), "Loss", loss.data[0])
        # tpn.print_weights()
        # loss.backward(retain_graph=True)
        optimizer.step()

        # if (epoch) % 100 == 0:
        #    print('Epoch [%d/%d], Loss: %.4f' % (epoch, num_epochs, loss.data[0]))

        tpn.spn.normalise_weights()
        #print('Epoch [%d/%d], Loss: %.4f' % (epoch, num_epochs, loss.data[0]))


##########################################################
# Test
##########################################################
print("\nTesting...\n")
# for i in range(4):
#    test = next(iter(train_loader))[0].float()
#    print("Input", test.numpy())
#    print("Prediction", tpn(test).detach().numpy())
# next(iter(train_loader))[1]

# tpn.print_weights()
print("Prob x1", torch.exp(tpn(x1)).cpu().detach().numpy())
print("Prob x2", torch.exp(tpn(x2)).cpu().detach().numpy())
print("Prob x3", torch.exp(tpn(x3)).cpu().detach().numpy())
print("Prob x4", torch.exp(tpn(x4)).cpu().detach().numpy())
print("Prob x5", torch.exp(tpn(x5)).cpu().detach().numpy())
print("Prob x6", torch.exp(tpn(x6)).cpu().detach().numpy())
print("Prob x7", torch.exp(tpn(x7)).cpu().detach().numpy())
print("Prob x8", torch.exp(tpn(x8)).cpu().detach().numpy())

print("weights")
tpn.print_weights()
