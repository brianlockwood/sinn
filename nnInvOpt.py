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
from heatSolve import primalSolve;
from heatSolve import adjointSolve;

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2,100) 
        self.fc2 = nn.Linear(100,1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
        
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
    
optimizer = optim.LBFGS(net.parameters())
criterion = nn.MSELoss()

Xobs = [[ 0.5555555555555556 ,  0.0 ],
        [ 0.5555555555555556 ,  0.1111111111111111 ],
        [ 0.5555555555555556 ,  0.2222222222222222 ],
        [ 0.5555555555555556 ,  0.3333333333333333 ],
        [ 0.5555555555555556 ,  0.4444444444444444 ],
        [ 0.5555555555555556 ,  0.5555555555555556 ],
        [ 0.5555555555555556 ,  0.6666666666666666 ],
        [ 0.5555555555555556 ,  0.7777777777777778 ],
        [ 0.5555555555555556 ,  0.8888888888888888 ],
        [ 0.5555555555555556 ,  1.0 ]]

Tobs= [299.99999999999073 ,
390.43449664981017 ,
494.36565681478646 ,
628.3731382140554 ,
690.101533275784 ,
690.1015332757842 ,
628.3731382140556 ,
494.36565681478714 ,
390.4344966498124 ,
299.999999999997 ]

Nobs = len(Tobs);

Xobs = torch.tensor(Xobs)
Tobs = torch.tensor(Tobs)
Tobs = Tobs.reshape(Nobs, 1)

T = np.ones(mesh.size())*300.0

# Traing to minimize residual
optimizer = optim.LBFGS(net.parameters())

Qi = torch.ones(mesh.size())*10000

for i in range(20):
    exitFlag = False
    def closure():
        optimizer.zero_grad()
        output = net(Xin)
        loss = criterion(output,Qi)
        print(loss)

        if (loss < 1e-2):
            exitFlag = True
        
        loss.backward()
        return loss
    
    if (not exitFlag):
        optimizer.step(closure)  

optimizer2 = optim.LBFGS(net.parameters())
     
for l in range(100):
    exitFlag = False
    def closure():
        optimizer2.zero_grad()
        Q = net(Xin)

        primalSolve(mesh,T,Tw,Q)
        Tdata = []
        i = 5
        for j in range(Ny):
            Tdata.append(T[mesh.ind(i,j)])

        Tdata2 = torch.tensor(Tdata, dtype=torch.float32, requires_grad = True)
        Tsim = Tdata2.reshape(Nobs, 1)

        loss = criterion(Tsim,Tobs)

        if (loss < 10):
            exitFlag = True
        
        loss.backward()
        dT = Tdata2.grad.detach().numpy()
        dLdT = np.zeros(mesh.size())
        for j in range(Ny):
            dLdT[mesh.ind(i,j)] = dT[j] 
        
        dTwall, dQ = adjointSolve(mesh, T, Tw, Q, dLdT)

        Q.backward(torch.tensor(dQ).reshape(Q.shape))

        print(loss)
        
        return loss

    if (not exitFlag):
        optimizer2.step(closure)

T = T.reshape(Nx,Ny)

Q = net(Xin)

print(T[5][:])
print(Q.reshape(Nx,Ny))

pyplot.contourf(X.reshape((Nx,Ny)),Y.reshape((Nx,Ny)),Q.view(Nx,Ny).detach().numpy())
pyplot.show()
