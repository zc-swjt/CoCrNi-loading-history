"""Matched K-amplitude histories for the first nonproportional crack pilot."""

import numpy as np

PATHS = {'P0': 30.0, 'P1': 0.0, 'P2': 60.0}
DT_PS = .001
STEPS = 84000


def eased_ramp(t, duration, rate=.025, ramp=2.):
    """Integrate a rate with cubic acceleration and deceleration ramps."""
    t = np.clip(np.asarray(t, dtype=float), 0., duration)
    s = np.clip(t/ramp, 0., 1.)
    first = rate*ramp*(s**3 - .5*s**4)
    middle = rate*(t - .5*ramp)
    s = np.clip((t - (duration-ramp))/ramp, 0., 1.)
    last = rate*(duration-1.5*ramp) + rate*ramp*(s-s**3+.5*s**4)
    return np.where(t < ramp, first, np.where(t <= duration-ramp, middle, last))


def history(path):
    angle = PATHS[path]
    t = np.arange(STEPS+1)*DT_PS
    amplitude = eased_ramp(t, 38.)
    amplitude[t >= 38.] = .9
    later = t >= 70.
    amplitude[later] = .9 + eased_ramp(t[later]-70., 14.)
    s = np.clip((t-38.)/30., 0., 1.)
    psi = angle + (30.-angle)*(3*s**2-2*s**3)
    psi[t >= 68.] = 30.
    rad = np.deg2rad(psi)
    return amplitude[:, None]*np.column_stack((np.cos(rad), np.sin(rad)))


def write_schedule(path, k):
    k = np.asarray(k, dtype=float)
    if k.ndim != 2 or k.shape[1] != 2 or not np.isfinite(k).all():
        raise ValueError('K history must be a finite N x 2 array')
    increments = np.diff(k, axis=0)
    np.savetxt(path, increments, fmt='%.17g',
               header=str(len(increments)), comments='')


def save_history(path, k):
    k = np.asarray(k)
    time = np.arange(len(k))*DT_PS
    amplitude = np.linalg.norm(k, axis=1)
    angle = np.rad2deg(np.arctan2(k[:, 1], k[:, 0]))
    np.savetxt(path, np.column_stack((time, k, amplitude, angle)), delimiter=',',
               fmt='%.12g', comments='',
               header='time_ps,KI_MPa_sqrt_m,KII_MPa_sqrt_m,K_MPa_sqrt_m,psi_deg')
