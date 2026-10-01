"""Computational functions used in the accompanying CoCrNi study."""
import numpy as np

def inverse_local_map(current, reference, queries, lz, neighbors=32):
    from scipy.spatial import cKDTree
    p = np.array(current, dtype=float, copy=True)
    r = np.array(reference, dtype=float, copy=True)
    q = np.array(queries, dtype=float, copy=True)
    r[:, 2] += np.rint((p[:, 2]-r[:, 2])/lz)*lz
    shift = np.floor(p[:, 2]/lz)*lz
    p[:, 2] -= shift
    r[:, 2] -= shift
    lift = np.floor(q[:, 2]/lz)*lz
    q[:, 2] -= lift
    shifts = np.array([[0, 0, -lz], [0, 0, 0], [0, 0, lz]])
    pp = np.concatenate([p+s for s in shifts])
    rr = np.concatenate([r+s for s in shifts])
    distance, index = cKDTree(pp).query(q, k=neighbors, workers=2)
    mapped, residual = [], []
    for target, d, ids in zip(q, distance, index):
        design = np.column_stack((np.ones(neighbors), pp[ids]-target))
        weights = 1/np.maximum(d, .5)**2
        w = np.sqrt(weights)
        coeff, _, rank, _ = np.linalg.lstsq(design*w[:, None], rr[ids]*w[:, None], rcond=None)
        if rank < 4:
            raise ValueError('Local inverse map lacks three-dimensional support')
        mapped.append(coeff[0])
        residual.append(np.sqrt(np.average(np.sum((design@coeff-rr[ids])**2, axis=1), weights=weights)))
    mapped = np.asarray(mapped)
    mapped[:, 2] += lift
    return mapped, np.asarray(residual)
