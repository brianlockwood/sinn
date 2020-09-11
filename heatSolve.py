from matplotlib import pyplot

import math;
import numpy as np;

from mesh import Mesh;
from heatResidual import residual;
from heatResidual import residual_b;
from heatResidual import residual_jac;


def primalSolve(mesh, T, Twall, Q, tol=1e-10, maxiter = 100):
            
    for i in range(maxiter):
                
        R = residual(mesh, T,Twall,Q)
        
        error = np.dot(R,R)
        error = math.sqrt(error/(1.0*mesh.size()))
        
        print(error)
        
        if (error < tol):
            break
        
        A = residual_jac(mesh, T, Twall, Q)
        
        dT = np.linalg.solve(A,R)

        T += dT


def adjointSolve(mesh, T, Twall, Q, dLdT, tol=1e-10, maxiter = 100):

    dQ = np.zeros(mesh.size())
    dTwall = 0.0

    Lambda = np.zeros(mesh.size())
    
    for i in range(maxiter):

        dT, dTwall, dQ = residual_b(mesh, T, Twall, Q, Lambda)

        R_b = dLdT + dT

        At = np.transpose(residual_jac(mesh, T, Twall, Q))

        dLam = np.linalg.solve(At,R_b)

        Lambda += dLam

    return dTwall, dQ

if __name__ == "__main__":
    Nx = 10
    Ny = 10
    
    Lx = 1.0
    Ly = 1.0
    
    mesh = Mesh(Nx,Ny,Lx,Ly)
    
    X,Y = mesh.generateCoordinates()
    
    T = np.ones(mesh.size())*350
    Twall = 300.0
    #Q = np.ones(mesh.size())*1000.0
    
    Q = np.ones(mesh.size())*10000.0
                
    primalSolve(mesh, T, Twall, Q)

    sum = 0.0
    for i in range(mesh.size()):
        sum += T[i]

    Tavg = sum/(1.0*mesh.size())

    print("L1=", Tavg)
    
    dLdT = np.ones(mesh.size())/(1.0*mesh.size()) 

    dTwall, dQ = adjointSolve(mesh, T, Twall, Q, dLdT)

    sum = 0.0
    for i in range(mesh.size()):
        sum += dQ[i]

    print("dLdQ = ", sum)
    

    Q = np.ones(mesh.size())*11000.0
    
    primalSolve(mesh, T, Twall, Q)

    sum = 0.0
    for i in range(mesh.size()):
        sum += T[i]

    Tavg2 = sum/(1.0*mesh.size())

    print("L2=",Tavg2)
    
    print("FD = ", (Tavg2-Tavg)/(11000.0-10000.0))
    
    # i = 5
    # for j in range(Ny):
    #     print("[", X[mesh.ind(i,j)], ", ",  Y[mesh.ind(i,j)], "],")

    # for j in range(Ny):
    #     print(T[mesh.ind(i,j)], ",")

    # X = X.reshape((Nx,Ny))
    # Y = Y.reshape((Nx,Ny))
    # T = T.reshape((Nx,Ny))
    # Q = Q.reshape((Nx,Ny))

    # print(Q)

    pyplot.contourf(X.reshape((Nx,Ny)),Y.reshape((Nx,Ny)),dQ.reshape((Nx,Ny)))
    pyplot.show()
    
