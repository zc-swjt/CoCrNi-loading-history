"""Computational functions used in the accompanying CoCrNi study."""
import numpy as np
N=445760

def state(file,weighted=False):
    a=np.load(file)
    known=np.zeros(N,bool); fault=np.zeros(N,bool); near=np.zeros(N,bool); weight=np.zeros(N)
    ids=a['atom_indices']
    known[ids]=True
    fault[ids]=a['fault_type']==2
    near[ids]=np.linalg.norm(a['positions_A'][:,:2]-200,axis=1)<50
    if weighted:
        weight[a['fault_atom_indices']]=a['fault_area_nm2']
    else:
        weight[ids]=1.
    return known,fault,near,weight

def budget(before,after):
    k0,f0,m0,w0=before; k1,f1,m1,w1=after
    both=k0&k1
    retained=both&f0&f1
    initial=float(w0[f0&m0].sum()); final=float(w1[f1&m1].sum())
    gain=float(w1[both&~f0&f1&m1].sum())
    loss=-float(w0[both&f0&~f1&m0].sum())
    inflow=float(w1[retained&~m0&m1].sum())
    outflow=-float(w0[retained&m0&~m1].sum())
    metric=float((w1-w0)[retained&m0&m1].sum())
    unknown=float(w1[~both&f1&m1].sum()-w0[~both&f0&m0].sum())
    residual=final-initial-gain-loss-inflow-outflow-metric-unknown
    assert abs(residual)<1e-8, residual
    return dict(initial=initial,final=final,net=final-initial,
        classification_gain=gain,classification_loss=loss,
        retained_fault_inflow=inflow,retained_fault_outflow=outflow,
        retained_area_metric_change=metric,unresolved_coverage=unknown,balance_error=residual)

def atom_state(a):
    n=len(a['fault_type'])
    return (np.ones(n,dtype=bool),a['fault_type']==2,
            np.linalg.norm(a['positions_A'][:,:2]-200,axis=1)<50,
            np.sqrt(3)/4*a['interatomic_distance_A']**2/100)

def fault_regions(positions, kinds, distances):
    r = np.linalg.norm(positions[:, :2]-200., axis=1)
    regions = dict(near=r<50., interior=(r>=50.) & (r<130.), outer=r>=130.,
                   total=np.ones(len(r), dtype=bool))
    row = {}
    for name, mask in regions.items():
        for kind, label, factor in ((2, 'ISF', np.sqrt(3)/4), (3, 'twin', np.sqrt(3)/2)):
            selected = mask & (kinds == kind)
            row[f'{label}_{name}_nm2'] = float(np.sum(factor*distances[selected]**2)/100)
    return row
