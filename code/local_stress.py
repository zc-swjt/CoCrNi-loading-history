"""Computational functions used in the accompanying CoCrNi study."""
import numpy as np
GPA=160.21766208
OMEGA=3.53721**3/4

def average_field(pos, virial, lz):
    edges = np.linspace(-50,50,21)
    xy=pos[:,:2]-200
    count=np.histogram2d(xy[:,0],xy[:,1],bins=(edges,edges))[0]
    sums=np.empty((20,20,3,3))
    sym=(virial+virial.transpose(0,2,1))/2
    for i in range(3):
        for j in range(3):
            sums[:,:,i,j]=np.histogram2d(xy[:,0],xy[:,1],bins=(edges,edges),weights=sym[:,i,j])[0]
    material=-sums/np.maximum(count[:,:,None,None],1)/OMEGA*GPA
    geometric=-sums/(25*lz)*GPA
    return dict(edges_A=edges, count=count, stress_material_GPa=material, stress_geometric_GPa=geometric)
