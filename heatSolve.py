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
Q = 1000.0

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
    

print(T.reshape((Nx,Ny)))

pyplot.contourf(X.reshape((Nx,Ny)),Y.reshape((Nx,Ny)),T.reshape((Nx,Ny)))
pyplot.show()
    
