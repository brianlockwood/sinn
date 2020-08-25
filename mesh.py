import numpy as np;

class Mesh: 
    def __init__(self,Nx,Ny,Lx,Ly):
        self.nx = Nx
        self.ny = Ny
        self.lx = Lx
        self.ly = Ly
        self.X  = []
        self.Y  = []

    def ind(self,i,j):
        return i*self.ny + j
    
    def size(self):
        return self.nx*self.ny

    def dimension(self):
        return self.nx, self.ny
    
    def generateCoordinates(self):

        nx = self.nx
        ny = self.ny
        
        X = np.zeros(self.size())
        Y = np.zeros(self.size())

        for i in range(nx):
            for j in range(ny):
                X[self.ind(i,j)] = 1.0*i/(1.0*(nx-1))*self.lx
                Y[self.ind(i,j)] = 1.0*j/(1.0*(ny-1))*self.ly

        self.X = X
        self.Y = Y
                
        return X,Y

    def getCoordinates(self):
        return self.X, self.Y
    
