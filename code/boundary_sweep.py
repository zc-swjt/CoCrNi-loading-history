"""Computational functions used in the accompanying CoCrNi study."""
import numpy as np
NORMALS=np.array([[1,1,1],[1,1,-1],[1,-1,1],[1,-1,-1]],float)/np.sqrt(3)
LABELS=["+++","++-","+-+","+--"]

def resample(p,count=65):
    p=np.asarray(p,float)
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    keep=np.r_[True,np.diff(arc)>1e-9]
    arc,p=arc[keep],p[keep]
    if len(p)<2:
        raise ValueError('Zero length curve')
    q=np.linspace(0,arc[-1],count)
    return np.column_stack([np.interp(q,arc,p[:,j]) for j in range(3)])

def align_periodic(a,b,lz):
    choices=[]
    for reverse in (False,True):
        q=b[::-1].copy() if reverse else b.copy()
        shift=float(np.rint(np.mean(a[:,2]-q[:,2])/lz)*lz)
        q[:,2]+=shift
        score=float(np.mean(np.sum((q-a)**2,axis=1)))
        choices.append((score,q,reverse,shift))
    _,q,reverse,shift=min(choices,key=lambda r:r[0])
    return q,reverse,shift

def ribbon(a,b,n):
    mid=(a+b)/2
    tangent=np.diff(mid,axis=0)
    delta=((b-a)[:-1]+(b-a)[1:])/2
    areas=np.einsum('ij,j->i',np.cross(tangent,delta),n)
    dl=np.linalg.norm(tangent,axis=1)
    return dict(signed_area=float(areas.sum()),length=float(dl.sum()),
                areas=areas,dl=dl,mid=(mid[:-1]+mid[1:])/2,
                tangent=tangent/np.maximum(dl[:,None],1e-12),delta=delta)

def chunks(mask):
    ids=np.flatnonzero(mask)
    return [v for v in np.split(ids,np.flatnonzero(np.diff(ids)>1)+1) if len(v)>=3]
