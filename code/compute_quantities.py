"""Compute stress, glide velocity and local virtual-work terms."""
import csv
from pathlib import Path
import numpy as np
from dislocation_energy_basis import C_VALUES, cubic_tensor, prelog_energy
from virtual_glide_balance import pk_force, polygon_energy, energy_derivative, area_derivative, AREA
from path_theory_baseline import nominal_tensor_modes

GAMMA_ISF_J_M2 = -0.05290915409588749


def actual_loading(csv_file, time_ps):
    h = np.genfromtxt(csv_file, delimiter=',', names=True)
    index = int(round(time_ps*1000))
    if not np.isclose(h['time_ps'][index], time_ps, atol=1e-9):
        raise ValueError('Requested time is not a saved loading step')
    return float(h['KI_MPa_sqrt_m'][index]), float(h['KII_MPa_sqrt_m'][index])


def sector_stress(positions_A, virial_eV, r_min_nm, r_max_nm, theta_min_deg=-30, theta_max_deg=30):
    xy = np.asarray(positions_A)[:, :2] - 200
    radius = np.linalg.norm(xy, axis=1)/10
    angle = np.rad2deg(np.arctan2(xy[:, 1], xy[:, 0]))
    mask = ((radius >= r_min_nm) & (radius < r_max_nm) &
            (angle >= theta_min_deg) & (angle < theta_max_deg))
    if not mask.any():
        raise ValueError('No atoms in the requested sector')
    v = np.asarray(virial_eV)[mask].sum(axis=0)
    return -(v+v.T)/2/mask.sum()/(3.53721**3/4)*160.21766208


def incremental_reference(stress0, unit_i, unit_ii, k, k0):
    dk = np.asarray(k)-np.asarray(k0)
    return np.asarray(stress0)+dk[0]*unit_i+dk[1]*unit_ii


def unit_sector_tensor(lo, hi, amin=-30, amax=30, mode=0, nr=100, nt=120):
    r = np.sqrt(lo**2+(np.arange(nr)+.5)/nr*(hi**2-lo**2))*10
    theta = np.deg2rad(amin+(np.arange(nt)+.5)/nt*(amax-amin))
    points = np.column_stack(((r[:, None]*np.cos(theta)).ravel()+200,
                             (r[:, None]*np.sin(theta)).ravel()+200,
                             np.zeros(nr*nt)))
    return nominal_tensor_modes(points, float(mode == 0), float(mode == 1)).mean(axis=0)


def glide_velocity(signed_area_nm2, participating_length_nm, interval_ps):
    return signed_area_nm2/participating_length_nm/interval_ps


def virtual_work(points_nm, mode, outward, burgers_nm, ki, kii, gamma=GAMMA_ISF_J_M2, pk_sign=-1):
    p, u, b = np.asarray(points_nm), np.asarray(mode), np.asarray(burgers_nm)
    delta = np.diff(p, axis=0)
    dl = np.linalg.norm(delta, axis=1)
    tangent = delta/dl[:, None]
    midpoint = (p[:-1]+p[1:])/2
    sigma = nominal_tensor_modes(midpoint*10, ki, kii)
    force = pk_force(sigma, b, tangent, pk_sign)
    qk = float(np.sum(np.einsum('ij,ij->i', force, (u[:-1]+u[1:])/2)*dl))*AREA
    elastic = cubic_tensor(*C_VALUES)
    energy = lambda x: polygon_energy(x, lambda t: prelog_energy(elastic, t, b)/1.602176634e-10)
    de = energy_derivative(p, u, energy)
    da = area_derivative(p, u, outward)
    qsf = -gamma*da*AREA
    result = {'Q_K_eV_nm': qk, 'Q_SF_eV_nm': qsf, 'dE_prelog_dq_eV_nm': de,
              'dA_dq_nm': da}
    for lam in (2, 3, 4):
        q = qk+qsf-lam*de
        result[f'Q_log{lam}_eV_nm'] = q
        result[f'f_log{lam}_N_m'] = q/da/AREA
    return result
