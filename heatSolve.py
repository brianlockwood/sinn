from matplotlib import pyplot

import math;
import numpy as np;

from mesh import Mesh;
from heatResidual import residual;
from heatResidual import residual_jac;
    
Nx = 10
Ny = 10
    
Lx = 1.0
Ly = 1.0

mesh = Mesh(Nx,Ny,Lx,Ly)

X,Y = mesh.generateCoordinates()

T = np.ones(mesh.size())*350
Twall = 300.0
#Q = np.ones(mesh.size())*1000.0

Q = np.zeros(mesh.size())

for i in range(Nx):
    for j in range(Ny):
        r = math.sqrt(math.pow(X[mesh.ind(i,j)]-Lx/2,2) + math.pow(Y[mesh.ind(i,j)]-Ly/2,2))
        Q[mesh.ind(i,j)] = 10000*math.sin(2*3.14159*r)

tol = 1e-10

for i in range(100):

    R = residual(mesh, T,Twall,Q)

    error = np.dot(R,R)
    error = math.sqrt(error/(1.0*mesh.size()))

    print(error)
    
    if (error < tol):
        break
    
    A = residual_jac(mesh, T, Twall, Q)

    dT = np.linalg.solve(A,R)

    T += dT

i = 5
print(T[mesh.ind(i,0):mesh.ind(i+1,0)])

X = X.reshape((Nx,Ny))
Y = Y.reshape((Nx,Ny))
T = T.reshape((Nx,Ny))
Q = Q.reshape((Nx,Ny))

print(T[5][:])

#pyplot.plot(Y[5][:], T[5][:])

print(Q)

pyplot.contourf(X.reshape((Nx,Ny)),Y.reshape((Nx,Ny)),Q.reshape((Nx,Ny)))
pyplot.show()
    
