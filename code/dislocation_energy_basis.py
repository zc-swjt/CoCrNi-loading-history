"""Computational functions used in the accompanying CoCrNi study."""
import numpy as np
C_VALUES=(290.7311626413,184.239628864,144.5485853796)

def cubic_tensor(c11,c12,c44):
    identity=np.eye(3)
    c=(c12*np.einsum('ij,kl->ijkl',identity,identity)
       +c44*(np.einsum('ik,jl->ijkl',identity,identity)+np.einsum('il,jk->ijkl',identity,identity)))
    for j in range(3): c[j,j,j,j]+=c11-c12-2*c44
    return c

def energy_tensor(c,line,n_angles=720):
    """Angular integral of (mm)-(mn)(nn)^-1(nm), in laboratory axes."""
    t=np.asarray(line,float); t=t/np.linalg.norm(t)
    axis=np.eye(3)[np.argmin(abs(t))]
    m0=np.cross(t,axis); m0/=np.linalg.norm(m0)
    n0=np.cross(t,m0)
    angle=2*np.pi*(np.arange(n_angles)+.5)/n_angles
    m=np.cos(angle)[:,None]*m0+np.sin(angle)[:,None]*n0
    n=-np.sin(angle)[:,None]*m0+np.cos(angle)[:,None]*n0
    nn=np.einsum('ai,ijkl,al->ajk',n,c,n)
    nm=np.einsum('ai,ijkl,al->ajk',n,c,m)
    mm=np.einsum('ai,ijkl,al->ajk',m,c,m)
    k=(mm-np.matmul(nm.transpose(0,2,1),np.linalg.solve(nn,nm))).mean(axis=0)
    return (k+k.T)/2

def prelog_energy(c,line,burgers_nm,n_angles=720):
    b=np.asarray(burgers_nm)*1e-9
    return float(b@energy_tensor(c,line,n_angles)@b)*1e9/(4*np.pi)
