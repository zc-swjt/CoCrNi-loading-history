"""Computational functions used in the accompanying CoCrNi study."""
import numpy as np
REGIONS=('near','interior','outer')

def regions(pos):
    r=np.linalg.norm(pos[:,:2]-200.,axis=1)
    return np.where(r<50,0,np.where(r<130,1,2))

def transition_rows(a,b):
    ka,kb=a['fault_type'].astype(int),b['fault_type'].astype(int)
    ra,rb=regions(a['positions_A']),regions(b['positions_A'])
    codes=np.column_stack((ka,kb,ra,rb))
    labels,inv=np.unique(codes,axis=0,return_inverse=True)
    count=np.bincount(inv)
    wa=np.sqrt(3)/4*a['interatomic_distance_A']**2/100
    wb=np.sqrt(3)/4*b['interatomic_distance_A']**2/100
    columns={
        'initial_ISF_nm2':np.bincount(inv,weights=wa*(ka==2)),
        'final_ISF_nm2':np.bincount(inv,weights=wb*(kb==2)),
        'initial_twin_nm2':np.bincount(inv,weights=2*wa*(ka==3)),
        'final_twin_nm2':np.bincount(inv,weights=2*wb*(kb==3))}
    return [dict(from_type=int(x),to_type=int(y),from_region=REGIONS[i],
                 to_region=REGIONS[j],atoms=int(count[k]),
                 **{name:float(v[k]) for name,v in columns.items()})
            for k,(x,y,i,j) in enumerate(labels)]
