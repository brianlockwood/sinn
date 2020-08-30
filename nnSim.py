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
        self.fc2 = nn.Linear(1000,2)

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
    def forward(ctx, T, mesh, Tw, Q, Tobs):
        ctx.save_for_backward(T,Q)
        ctx.mesh = mesh
        ctx.Tw = Tw
        ctx.Tobs = Tobs

        sum = torch.tensor(0.0, requires_grad=True)
        
        R = residual(mesh, T, Tw, Q)
       
        ctx.R = R

        sum = torch.tensor(0.0, requires_grad=True)       
        for i in range(mesh.size()):
            sum += R[i]*R[i]

        Nx, Ny = mesh.dimension()
        i = 5
        for j in range(Ny):
            sum += torch.dot((T[mesh.ind(i,j)] - Tobs[j]),(T[mesh.ind(i,j)] - Tobs[j]))
            
        return sum 

    @staticmethod
    def backward(ctx, grad_output):
        T,Q = ctx.saved_tensors
        
        mesh = ctx.mesh
        Tw = ctx.Tw
        R = ctx.R
        Tobs = ctx.Tobs
        
        grad_T = None

        dR = torch.zeros(mesh.size())
                       
        for i in range(mesh.size()):
            dR[i] = 2.0*R[i]*grad_output      

        dT,dTw,dQ = residual_b(mesh,T,Tw,Q,dR)

        Nx, Ny = mesh.dimension()
        i = 5
        for j in range(Ny):
            dT[mesh.ind(i,j)] += 2.0*(T[mesh.ind(i,j)] - Tobs[j])
            
        grad_T = torch.tensor(dT.reshape((mesh.size(),1)))
        grad_Q = torch.tensor(dQ.reshape((mesh.size(),1)))
        
        return grad_T, None, None, grad_Q, None
        
net = Net()
print(net)

Nx = 10
Ny = 10

Lx = 1.0
Ly = 1.0

mesh = Mesh(Nx,Ny,Lx,Ly)
Tw = 300.0

X,Y = mesh.generateCoordinates()

Xin = torch.zeros(mesh.size(),2)

l = 0
for k in range(mesh.size()):
    entry = torch.tensor([X[k], Y[k]])
    Xin[l] = entry
    l = l + 1
            
#print(Xin)

Tinitial = torch.ones(mesh.size(),1)*300

Tobs = torch.tensor([300.,         522.53657591, 698.79994668, 797.78856659, 827.03832688,
                     827.03832688, 797.78856659, 698.79994668, 522.53657591, 300.        ])

i = 5
Tinitial[mesh.ind(i,0):mesh.ind(i+1,0)]= Tobs.reshape(10,1)

print(Tinitial)

Qinitial = torch.ones(mesh.size(),1)*10000

optimizer = optim.LBFGS(net.parameters())
criterion = nn.MSELoss()

# Train Network to reproduce initial condition
for i in range(100):
    def closure():
        optimizer.zero_grad()
        output = net(Xin)
        T = torch.narrow(output,1,0,1)
        Q = torch.narrow(output,1,1,1)
        loss = criterion(T,Tinitial)
        loss += criterion(Q,Qinitial)
        print(loss)
        loss.backward()
        return loss
    optimizer.step(closure)  

# Traing to minimize residual
optimizer2 = optim.LBFGS(net.parameters())
for i in range(500):
    def closure():
        optimizer2.zero_grad()
        output = net(Xin)
        T = torch.narrow(output,1,0,1)
        Q = torch.narrow(output,1,1,1)
        loss = ResidualLoss.apply(T, mesh, Tw, Q, Tobs)
        print(loss)
        loss.backward()
        return loss
    optimizer2.step(closure)

output = net(Xin)
T = torch.narrow(output,1,0,1)
Q = torch.narrow(output,1,1,1)

T = T.reshape(Nx,Ny)

print(T[5][:])
print(Q.reshape(Nx,Ny))

pyplot.contourf(X.reshape((Nx,Ny)),Y.reshape((Nx,Ny)),Q.view(Nx,Ny).detach().numpy())
pyplot.show()
