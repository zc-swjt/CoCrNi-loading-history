"""Computational functions used in the accompanying CoCrNi study."""
import numpy as np
from slip_plane_audit import periodic_tree

def global_direction(vector):
    u=np.asarray(vector,float); u=u/np.linalg.norm(u)
    h=np.rint(u*np.sqrt(6)).astype(int)
    if not np.array_equal(np.sort(abs(h)),[1,1,2]):
        raise ValueError('Global direction is not close to <112>')
    angle=float(np.degrees(np.arccos(np.clip(u@(h/np.linalg.norm(h)),-1,1))))
    if angle>12:
        raise ValueError('Large orientation deviation')
    if h[np.flatnonzero(h)[0]]<0: h=-h
    return ','.join(map(str,h)),angle

def sample_curve(points,spacing=.5):
    p=np.asarray(points,float)
    samples,weights=[],[]
    for a,b in zip(p[:-1],p[1:]):
        length=float(np.linalg.norm(b-a))
        if length<1e-10: continue
        n=max(1,int(np.ceil(length/spacing)))
        samples.append(a+((np.arange(n)+.5)/n)[:,None]*(b-a))
        weights.append(np.full(n,length/n))
    if not samples: return np.zeros((0,3)),np.zeros(0)
    return np.concatenate(samples),np.concatenate(weights)

def near_samples(line):
    p,w=sample_curve(line['points_A'])
    mask=np.linalg.norm(p[:,:2]-200,axis=1)<50
    return p[mask],w[mask]

def distances(points,reference,lz):
    if len(reference)==0: return np.full(len(points),np.inf)
    p=np.array(points,copy=True); p[:,2]%=lz
    return periodic_tree(reference,lz).query(p,workers=2)[0]

def curve_links(previous,current,lz,cutoff_A=4.):
    old=[near_samples(x) for x in previous]
    new=[near_samples(x) for x in current]
    edges=[]; total=sum(w.sum() for _,w in new); matched=0.
    ds_all=[]; weights_all=[]
    for j,((q,w),line) in enumerate(zip(new,current)):
        if not len(q): continue
        all_ref=[old[i][0] for i,x in enumerate(previous)
                 if x['direction']==line['direction'] and len(old[i][0])]
        d=distances(q,np.concatenate(all_ref) if all_ref else np.zeros((0,3)),lz)
        matched+=w[d<=cutoff_A].sum()
        ds_all.append(d); weights_all.append(w)
        for i,((p,v),parent) in enumerate(zip(old,previous)):
            if parent['direction']!=line['direction'] or not len(p): continue
            d1,d2=distances(q,p,lz),distances(p,q,lz)
            frac1=float(w[d1<=cutoff_A].sum()/w.sum())
            frac2=float(v[d2<=cutoff_A].sum()/v.sum())
            if max(frac1,frac2)<.05: continue
            edges.append(dict(previous_segment=parent['segment_id'],current_segment=line['segment_id'],
                direction=line['direction'],current_to_previous_mean_A=float(np.average(d1,weights=w)),
                previous_to_current_mean_A=float(np.average(d2,weights=v)),
                current_near_parent_fraction=frac1,previous_near_child_fraction=frac2))
    d=np.concatenate(ds_all) if ds_all else np.array([])
    w=np.concatenate(weights_all) if weights_all else np.array([])
    finite=np.isfinite(d)
    summary=dict(current_near_length_nm=float(total/10),
        matched_current_fraction=float(matched/total) if total else 0.,
        current_length_without_same_direction_predecessor_nm=float(w[~finite].sum()/10),
        weighted_nearest_curve_distance_A=float(np.average(d[finite],weights=w[finite])) if finite.any() else None)
    return edges,summary
