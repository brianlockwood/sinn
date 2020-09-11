import nlopt

from matplotlib import pyplot
from matplotlib import cm

import numpy as np;
import math;

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as Grad

import torch.optim as optim

from mesh import Mesh;
from heatResidual import residual;
from heatResidual import residual_b;

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1,100)
        self.fc2 = nn.Linear(100,1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def func(x, grad, net, Xin, Yobs):
    size = 0
    ind = []
    for param in net.parameters():
        ind.append(size)
        size += param.nelement()
        
    ind.append(size)
    
    i = 0
    for param in net.parameters():
        paramSize = param.data.shape
        xsub = np.copy(x[ind[i]:ind[i+1]])
        param.data.copy_(torch.tensor(xsub, dtype=torch.float64).reshape(paramSize))
        
        i += 1

    Y = net(Xin)

    criterion = nn.MSELoss()

    loss = criterion(Y, Yobs)

    if grad.size > 0:
        for param in net.parameters():
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

        loss.backward()

        i = 0
        for param in net.parameters():
            gradSub = param.grad.reshape(param.grad.nelement()).detach().numpy()
            grad[ind[i]:ind[i+1]] = gradSub
            i += 1
            
    print(loss.detach().numpy())
        
    return float(loss.detach().numpy())
        
net = Net()
print(net)

size = 0
for param in net.parameters():
    size += param.nelement()

Xin = torch.linspace(0,1,10).reshape(10,1)
Yobs = torch.zeros(10,1)

for i in range(10):
    Yobs[i] = Xin[i] + math.cos(2*3.14159*Xin[i]/0.1)

print(size)

opt = nlopt.opt(nlopt.LD_SLSQP, size)

lb = []
ub = []
for i in range(size):
    lb.append(-float('inf'))
    ub.append(float('inf'))


opt.set_lower_bounds(lb)
opt.set_upper_bounds(ub)
opt.set_min_objective(lambda x,grad: func(x,grad,net,Xin,Yobs))
opt.set_maxeval(1000)

xi = []

for i in range(size):
    print(i)
    xi.append(0.0)

x = opt.optimize(xi)

size = 0
ind = []
for param in net.parameters():
    ind.append(size)
    size += param.nelement()
    
ind.append(size)
     
i = 0
for param in net.parameters():
    paramSize = param.data.shape
    xsub = np.copy(x[ind[i]:ind[i+1]])
    
    param.data.copy_(torch.tensor(xsub, dtype=torch.float64).reshape(paramSize))
    i += 1
    
Y = net(Xin)

print(Yobs)

print(Y)
