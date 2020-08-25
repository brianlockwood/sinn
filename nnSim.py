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
        self.fc1 = nn.Linear(2,1000) 
        self.fc2 = nn.Linear(1000,1)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class ResidualLoss(Grad.Function):
    
    @staticmethod
    def forward(ctx, T, mesh, Tw, Q):
        ctx.save_for_backward(T)
        ctx.mesh = mesh
        ctx.Tw = Tw
        ctx.Q = Q

        sum = torch.tensor(0.0, requires_grad=True)

        R = residual(mesh, T, Tw, Q)
       
        ctx.R = R

        sum = torch.tensor(0.0, requires_grad=True)       
        for i in range(mesh.size()):
            sum += R[i]*R[i]

        return torch.sqrt(sum/(1.0*mesh.size()))

    @staticmethod
    def backward(ctx, grad_output):
        T, = ctx.saved_tensors
        
        mesh = ctx.mesh
        Tw = ctx.Tw
        Q = ctx.Q
        R = ctx.R
        
        grad_T = None

        dR = torch.zeros(mesh.size())

        sum = 0.0
        for i in range(mesh.size()):
            sum += R[i]*R[i]
                  
        fac = 1.0/(2.0*math.sqrt(sum/(1.0*mesh.size()))*mesh.size())       
        
        for i in range(mesh.size()):
            dR[i] = 2.0*R[i]*grad_output*fac       
            
        dT,dTw,dQ = residual_b(mesh,T,Tw,Q,dR)

        grad_T = torch.tensor(dT.reshape((mesh.size(),1)))

        return grad_T, None, None, None
        
net = Net()
print(net)

Nx = 10
Ny = 10

Lx = 1.0
Ly = 1.0

mesh = Mesh(Nx,Ny,Lx,Ly)
Tw = 300.0
Q = 1000.0

X,Y = mesh.generateCoordinates()

Xin = torch.zeros(mesh.size(),2)

l = 0
for k in range(mesh.size()):
    entry = torch.tensor([X[k], Y[k]])
    Xin[l] = entry
    l = l + 1
            
#print(Xin)

Tinitial = torch.ones(mesh.size(),1)*300

optimizer = optim.LBFGS(net.parameters())
criterion = nn.MSELoss()


# Train Network to reproduce initial condition
for i in range(20):
    def closure():
        optimizer.zero_grad()
        output = net(Xin)
        loss = criterion(output,Tinitial)
        print(loss)
        loss.backward()
        return loss
    optimizer.step(closure)


# Traing to minimize residual
optimizer2 = optim.LBFGS(net.parameters())
for i in range(20):
    def closure():
        optimizer2.zero_grad()
        output = net(Xin)
        TwOutput = net(Xin)
        loss = ResidualLoss.apply(output,mesh, Tw, Q)
        print(loss)
        loss.backward()
        return loss
    optimizer2.step(closure)

T = net(Xin)

pyplot.contourf(X.reshape((Nx,Ny)),Y.reshape((Nx,Ny)),T.view(Nx,Ny).detach().numpy())
pyplot.show()
