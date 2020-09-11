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
    def forward(ctx, T, mesh, Tw, Q):
        ctx.save_for_backward(T,Q)
        ctx.mesh = mesh
        ctx.Tw = Tw

        sum = torch.tensor(0.0, requires_grad=True)
        
        R = residual(mesh, T, Tw, Q)
       
        ctx.R = R

        sum = torch.tensor(0.0, requires_grad=True)       
        for i in range(mesh.size()):
            sum += R[i]*R[i]
            
        return sum 

    @staticmethod
    def backward(ctx, grad_output):
        T,Q = ctx.saved_tensors
        
        mesh = ctx.mesh
        Tw = ctx.Tw
        R = ctx.R
        
        grad_T = None

        dR = torch.zeros(mesh.size())
                       
        for i in range(mesh.size()):
            dR[i] = 2.0*R[i]*grad_output      

        dT,dTw,dQ = residual_b(mesh,T,Tw,Q,dR)
            
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
            
Tinitial = torch.ones(mesh.size(),1)*300

Qinitial = torch.ones(mesh.size(),1)*10000

for i in range(Nx):
    for j in range(Ny):
        r = math.sqrt(math.pow(X[mesh.ind(i,j)]-Lx/2,2) + math.pow(Y[mesh.ind(i,j)]-Ly/2,2))
        Qinitial[mesh.ind(i,j)] = 10000*math.sin(2*3.14159*r)



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

Tobs= [299.99999999997146 ,
       522.5365759112073 ,
       698.7999466754953 ,
       797.788566591018 ,
       827.0383268786188 ,
       827.0383268786195 ,
       797.7885665910198 ,
       698.7999466754993 ,
       522.5365759112182 ,
       300.0000000000094 ]

Nobs = len(Tobs);

Xobs = torch.tensor(Xobs)
Tobs = torch.tensor(Tobs)
Tobs = Tobs.reshape(Nobs, 1)

print(Xobs)
print(Xin.shape)
print(Xobs.shape)
print(Tobs.shape)


# Train Network to reproduce initial condition
for i in range(1):
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
for i in range(100):
    def closure():
        optimizer2.zero_grad()
        output = net(Xin)
        T = torch.narrow(output,1,0,1)
        Q = torch.narrow(output,1,1,1)
        loss = ResidualLoss.apply(T, mesh, Tw, Q)
        output2 = net(Xobs)
        T = torch.narrow(output2,1,0,1)
        loss += criterion(T, Tobs)        
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
