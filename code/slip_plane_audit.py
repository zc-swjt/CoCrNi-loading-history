"""Computational functions used in the accompanying CoCrNi study."""
import numpy as np
import itertools
NORMALS=np.array([[1,1,1],[1,1,-1],[1,-1,1],[1,-1,-1]],float)/np.sqrt(3)
LABELS=["+++","++-","+-+","+--"]

def periodic_tree(points, lz):
    from scipy.spatial import cKDTree
    p = np.array(points, float, copy=True)
    p[:,2] %= lz
    copies = np.concatenate([p + [0,0,z] for z in (-lz,0,lz)])
    return cKDTree(copies)

def classify_planes(reference_points, lz, radius=5.5):
    from scipy.spatial import cKDTree
    p = np.array(reference_points, float, copy=True)
    p[:,2] %= lz
    tree = periodic_tree(p, lz)
    neighbors = tree.query_ball_point(p, radius, workers=2)
    labels = np.full(len(p), 'unresolved', dtype='<U10')
    angles, ratios = np.full(len(p), np.nan), np.full(len(p), np.nan)
    for i, ids in enumerate(neighbors):
        if len(ids) < 8:
            continue
        local = tree.data[ids]
        q = local - local.mean(axis=0)
        values, vectors = np.linalg.eigh(q.T @ q / len(q))
        if values[1] < .2:
            continue
        cosines = np.abs(NORMALS @ vectors[:,0])
        j = int(cosines.argmax())
        angles[i] = np.degrees(np.arccos(np.clip(cosines[j], 0, 1)))
        ratios[i] = max(values[0], 0) / values[1]
        if angles[i] <= 20 and ratios[i] <= .45:
            labels[i] = LABELS[j]
    return labels, angles, ratios

def canonical_b(b):
    h = np.rint(np.array(b) * 6).astype(int)
    if not np.allclose(h, np.array(b)*6, atol=.02):
        raise ValueError('Not a crystallographic a/6 Burgers vector')
    if h[np.flatnonzero(h)[0]] < 0:
        h = -h
    return ','.join(map(str, h))

def slip_directions(n):
    out = []
    for p in sorted(set(itertools.permutations((1,1,2)))):
        for signs in itertools.product((-1,1), repeat=3):
            b = np.array(p) * signs
            if b[0] < 0 or abs(b @ n) > 1e-8:
                continue
            out.append(b)
    assert len(out) == 3
    return out
