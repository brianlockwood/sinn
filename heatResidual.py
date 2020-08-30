from matplotlib import pyplot

import math;
import numpy as np;

from mesh import Mesh;

from scipy import optimize;

def residual(mesh, T, Twall, Q):

    Nx, Ny = mesh.dimension()

    X,Y = mesh.getCoordinates();

    ind = mesh.ind
    
    R = np.zeros(mesh.size())

    i = 0
    for j in range(Ny):
        R[ind(i,j)] = Twall-T[ind(i,j)]
        
    i = Nx-1
    for j in range(Ny):
        R[ind(i,j)] = Twall-T[ind(i,j)]

    j = 0
    for i in range(1,Nx-1):
        R[ind(i,j)] = Twall-T[ind(i,j)]

    j = Ny-1
    for i in range(1,Nx-1):
        R[ind(i,j)] = Twall-T[ind(i,j)]


    for i in range(1,Nx-1):
        for j in range(1,Ny-1):
            dXL = (X[ind(i-1,j)] - X[ind(i,j)])
            dXR = (X[ind(i+1,j)] - X[ind(i,j)])

            dYU = (Y[ind(i,j+1)] - Y[ind(i,j)])
            dYD = (Y[ind(i,j-1)] - Y[ind(i,j)])

            dTxL = (T[ind(i-1,j)] - T[ind(i,j)])/dXL
            dTxR = (T[ind(i+1,j)] - T[ind(i,j)])/dXR
            dTyU = (T[ind(i,j+1)] - T[ind(i,j)])/dYU
            dTyD = (T[ind(i,j-1)] - T[ind(i,j)])/dYD

            R[ind(i,j)] = 2.0*(dTxR - dTxL)/(dXR-dXL) + 2.0*(dTyU - dTyD)/(dYU-dYD) + Q[ind(i,j)]

    return R

def residual_jac(mesh, T, Twall, Q):

    Nx, Ny = mesh.dimension()

    X,Y = mesh.getCoordinates();

    ind = mesh.ind
    
    A = np.zeros((mesh.size(), mesh.size()))

    i = 0
    for j in range(Ny):
        A[ind(i,j),ind(i,j)] = 1.0
        
    i = Nx-1
    for j in range(Ny):
        A[ind(i,j),ind(i,j)] = 1.0

    j = 0
    for i in range(1,Nx-1):
        A[ind(i,j),ind(i,j)] = 1.0
        
    j = Ny-1
    for i in range(1,Nx-1):
        A[ind(i,j),ind(i,j)] = 1.0

    for i in range(1,Nx-1):
        for j in range(1,Ny-1):
            dXL = (X[ind(i-1,j)] - X[ind(i,j)])
            dXR = (X[ind(i+1,j)] - X[ind(i,j)])

            dYU = (Y[ind(i,j+1)] - Y[ind(i,j)])
            dYD = (Y[ind(i,j-1)] - Y[ind(i,j)])

            A[ind(i,j),ind(i,j)] = -2.0*(-1/dXR + 1/dXL)/(dXR-dXL) - 2.0*(-1/dYU+1/dYD)/(dYU-dYD)
            A[ind(i,j),ind(i+1,j)] = -2.0*(1/dXR)/(dXR-dXL)
            A[ind(i,j),ind(i-1,j)] = 2.0*(1/dXL)/(dXR-dXL)
            A[ind(i,j),ind(i,j+1)] = -2.0*(1/dYU)/(dYU-dYD)
            A[ind(i,j),ind(i,j-1)] = 2.0*(1/dYD)/(dYU-dYD)

    return A

def residual_d(mesh, T, Twall, Q, dT, dTwall, dQ):
    dR = np.zeros(mesh.size())

    Nx, Ny = mesh.dimension()

    X,Y = mesh.getCoordinates();

    ind = mesh.ind
    
    i = 0
    for j in range(Ny):
        dR[ind(i,j)] += dTwall-dT[ind(i,j)]
        
    i = Nx-1
    for j in range(Ny):
        dR[ind(i,j)] += dTwall-dT[ind(i,j)]
        
    j = 0
    for i in range(1,Nx-1):
        dR[ind(i,j)] += dTwall-dT[ind(i,j)]

    j = Ny-1
    for i in range(1,Nx-1):
        dR[ind(i,j)] += dTwall-dT[ind(i,j)]
        
    for i in range(1,Nx-1):
        for j in range(1,Ny-1):
            dXL = (X[ind(i-1,j)] - X[ind(i,j)])
            dXR = (X[ind(i+1,j)] - X[ind(i,j)])

            dYU = (Y[ind(i,j+1)] - Y[ind(i,j)])
            dYD = (Y[ind(i,j-1)] - Y[ind(i,j)])

            dTxL = (T[ind(i-1,j)] - T[ind(i,j)])/dXL
            dTxL_d = (dT[ind(i-1,j)] - dT[ind(i,j)])/dXL
            dTxR = (T[ind(i+1,j)] - T[ind(i,j)])/dXR
            dTxR_d = (dT[ind(i+1,j)] - dT[ind(i,j)])/dXR
            dTyU = (T[ind(i,j+1)] - T[ind(i,j)])/dYU
            dTyU_d = (dT[ind(i,j+1)] - dT[ind(i,j)])/dYU
            dTyD = (T[ind(i,j-1)] - T[ind(i,j)])/dYD
            dTyD_d = (dT[ind(i,j-1)] - dT[ind(i,j)])/dYD

            dR[ind(i,j)] = 2.0*(dTxR_d - dTxL_d)/(dXR-dXL) + 2.0*(dTyU_d - dTyD_d)/(dYU-dYD) + dQ[ind(i,j)]
            
    return dR

def residual_b(mesh, T, Twall, Q, dR):
    dT = np.zeros(mesh.size())
    dTwall = np.zeros(1);
    dQ = np.zeros(mesh.size());
    
    Nx, Ny = mesh.dimension()

    X,Y = mesh.getCoordinates();

    ind = mesh.ind
    
    i = 0
    for j in range(Ny):
        dT[ind(i,j)] += -dR[ind(i,j)]
        dTwall[0] += dR[ind(i,j)]
        
    i = Nx-1
    for j in range(Ny):
        dT[ind(i,j)] += -dR[ind(i,j)]
        dTwall[0] += dR[ind(i,j)]
        
    j = 0
    for i in range(1,Nx-1):
        dT[ind(i,j)] += -dR[ind(i,j)]
        dTwall[0] += dR[ind(i,j)]

    j = Ny-1
    for i in range(1,Nx-1):
        dT[ind(i,j)] += -dR[ind(i,j)]
        dTwall[0] += dR[ind(i,j)]
       
    for i in range(1,Nx-1):
        for j in range(1,Ny-1):
            dXL = (X[ind(i-1,j)] - X[ind(i,j)])
            dXR = (X[ind(i+1,j)] - X[ind(i,j)])

            dYU = (Y[ind(i,j+1)] - Y[ind(i,j)])
            dYD = (Y[ind(i,j-1)] - Y[ind(i,j)])

            dTxL = (T[ind(i-1,j)] - T[ind(i,j)])/dXL
            dTxR = (T[ind(i+1,j)] - T[ind(i,j)])/dXR
            dTyU = (T[ind(i,j+1)] - T[ind(i,j)])/dYU
            dTyD = (T[ind(i,j-1)] - T[ind(i,j)])/dYD

            dTxR_b = 0.0
            dTxL_b = 0.0

            dTyU_b = 0.0
            dTyD_b = 0.0
            
            dTxR_b += 2.0/(dXR-dXL)*dR[ind(i,j)]
            dTxL_b += -2.0/(dXR-dXL)*dR[ind(i,j)]

            dTyU_b += 2.0/(dYU-dYD)*dR[ind(i,j)]
            dTyD_b += -2.0/(dYU-dYD)*dR[ind(i,j)]
            
            dT[ind(i-1,j)] += dTxL_b/dXL
            dT[ind(i,j)]   += -dTxL_b/dXL
            
            dT[ind(i+1,j)] += dTxR_b/dXR
            dT[ind(i,j)]   += -dTxR_b/dXR

            dT[ind(i,j+1)] += dTyU_b/dYU
            dT[ind(i,j)]   += -dTyU_b/dYU
            
            dT[ind(i,j-1)] += dTyD_b/dYD
            dT[ind(i,j)]   += -dTyD_b/dYD

            dQ[ind(i,j)] += dR[ind(i,j)]
            
    return dT, dTwall, dQ
    

if __name__ == "__main__":
    Nx = 5
    Ny = 5
    
    Lx = 1.0
    Ly = 1.0

    mesh = Mesh(Nx,Ny,Lx,Ly)
    
    X,Y = mesh.generateCoordinates()
    
    T = np.ones(mesh.size())*350
    Twall = np.ones(1)*300.0
    Q = np.ones(mesh.size())*1000.0

    dRin = np.random.rand(mesh.size())
    dTin = np.random.rand(mesh.size())
    dTwall = np.random.rand(1)
    dQ = np.random.rand(mesh.size())

    eps = 1e-2
    
    T1 = T - eps*dTin
    Twall1 = Twall - eps*dTwall
    Q1 = Q - eps*dQ
    
    R1 = residual(mesh,T1, Twall1, Q1)

    T2 = T + eps*dTin
    Twall2 = Twall + eps*dTwall
    Q2 = Q + eps*dQ

    R2 = residual(mesh,T2, Twall2, Q2)

    dR_fd = (R2-R1)/(2.0*eps)
    
    dRout = residual_d(mesh, T, Twall, Q, dTin, dTwall, dQ)
    
    err = 0.0
    for i in range(mesh.size()):
        err = max(abs(dR_fd[i]-dRout[i]), err)

    print(err)
        
    dTout, dTwOut, dQOut = residual_b(mesh, T, Twall, Q, dRin)

    sum1 = np.dot(dRout, dRin)
    sum2 = np.dot(dTout, dTin)
    sum2 += np.dot(dTwOut,dTwall)
    sum2 += np.dot(dQ,dQOut)
    
    print(sum1, sum2)
