"""Computational functions used in the accompanying CoCrNi study."""
import numpy as np
from slip_plane_audit import NORMALS,LABELS
from track_reload_slip import sample_curve
EV=1.602176634e-19
AREA=1e-18/EV

def pk_force(stress_GPa,b_nm,tangent,multiplier):
    return multiplier*np.cross(np.einsum('aij,j->ai',stress_GPa,b_nm),tangent)

def polygon_energy(p_nm,coefficient):
    d=np.diff(p_nm,axis=0); lengths=np.linalg.norm(d,axis=1)
    return sum(l*coefficient(v/l) for v,l in zip(d,lengths) if l>1e-12)

def energy_derivative(p,u,energy,h=.001):
    return (energy(p+h*u)-energy(p-h*u))/(2*h)

def area_derivative(p,u,outward):
    delta=np.diff(p,axis=0)
    normal=np.cross((outward[:-1]+outward[1:])/2,delta)
    normal/=np.linalg.norm(normal,axis=1)[:,None]
    swept=np.cross((u[:-1]+u[1:])/2,delta)
    return float(np.sum(swept*normal))

def chunks(mask):
    ids=np.flatnonzero(mask)
    return [x for x in np.split(ids,np.flatnonzero(np.diff(ids)>1)+1) if len(x)>=5]

def mode(line,a):
    from scipy.spatial import cKDTree
    p=line['points_nm']; n=line['normal']
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    if arc[-1]<1 or np.any(np.diff(arc)<1e-8): return None
    t=np.gradient(p,arc,axis=0)
    t/=np.linalg.norm(t,axis=1)[:,None]
    side=np.cross(n,t); side/=np.linalg.norm(side,axis=1)[:,None]
    idx=np.searchsorted(a['atom_indices'],a['fault_atom_indices'])
    faults=a['reference_positions_A'][idx]/10
    faults=faults[a['fault_plane']==LABELS[line['plane_index']]]
    if len(faults)<8: return None
    lz=float(a['cell_z_A'])/10
    faults=np.concatenate([faults+np.array([0,0,k*lz]) for k in range(-3,4)])
    dist,ids=cKDTree(faults).query(p,k=8)
    neighbor=faults[ids]
    same_plane=abs(np.einsum('aij,j->ai',neighbor-p[:,None,:],n))<.16
    weights=same_plane/(dist+.025)**2
    valid=weights.sum(axis=1)>0
    means=np.sum(neighbor*weights[:,:,None],axis=1)/np.maximum(weights.sum(axis=1)[:,None],1e-12)
    scores=np.einsum('ij,ij->i',p-means,side)
    radial=np.linalg.norm(p[:,:2]-20,axis=1)
    taper=np.sin(np.pi*arc/arc[-1])**2
    taper*=np.sin(np.pi/2*np.clip((radial-1)/.5,0,1))**2
    taper*=np.sin(np.pi/2*np.clip((5-radial)/.5,0,1))**2
    eligible=valid&(taper>.1)&(dist[:,0]<.8)
    if eligible.sum()<3: return None
    orientation=1 if np.median(scores[eligible])>=0 else -1
    outward=orientation*side
    u=outward*taper[:,None]
    u[0]=u[-1]=0
    fraction=float(np.mean(orientation*scores[eligible]>0))
    return u,outward,fraction,float(np.median(orientation*scores[eligible]))

def observed_motion(line,u,outward,next_lines,lz):
    from scipy.spatial import cKDTree
    references=[]
    for candidate in next_lines:
        if candidate['direction']!=line['direction']: continue
        samples,_=sample_curve(candidate['points_nm'],spacing=.025)
        references.append(samples)
    if not references: return None,None
    ref=np.concatenate(references)
    ref[:,2]%=lz
    copies=np.concatenate([ref+np.array([0,0,k*lz]) for k in (-1,0,1)])
    q=line['points_nm'].copy(); q[:,2]%=lz
    distance,idx=cKDTree(copies).query(q)
    shift=copies[idx]-q
    w=np.linalg.norm(u,axis=1)
    active=w>.05
    if not active.any(): return None,None
    signed=np.einsum('ij,ij->i',shift,outward)
    return float(np.average(signed[active],weights=w[active])),float(np.average(distance[active],weights=w[active]))
