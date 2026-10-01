"""Focused checks of the numerical functions, without MD or plotting."""
import numpy as np
from schedule import history
from dislocation_energy_basis import cubic_tensor, prelog_energy
from fault_area import budget
from boundary_sweep import resample, align_periodic, ribbon
from virtual_glide_balance import polygon_energy, energy_derivative, area_derivative
from compute_quantities import sector_stress, glide_velocity, incremental_reference
from local_stress import average_field
from structure_transitions import transition_rows
from fault_area import fault_regions


def main():
    for path in ('P1', 'P2'):
        np.testing.assert_allclose(history(path)[70000:], history('P0')[70000:], atol=1e-15)
    mu, nu, b = 50., .3, .2
    lam = 2*mu*nu/(1-2*nu)
    c = cubic_tensor(lam+2*mu, lam, mu)
    screw = prelog_energy(c, [0, 0, 1], [0, 0, b])
    edge = prelog_energy(c, [0, 0, 1], [b, 0, 0])
    np.testing.assert_allclose(screw, mu*1e9*(b*1e-9)**2/(4*np.pi), rtol=1e-12)
    np.testing.assert_allclose(edge/screw, 1/(1-nu), rtol=1e-12)
    p = np.column_stack((np.zeros(9), np.linspace(0, 2, 9), np.zeros(9)))
    q = p + [1, 0, 0]
    r = ribbon(p, q, [0, 0, -1])
    np.testing.assert_allclose(r['signed_area'], 2.)
    np.testing.assert_allclose(glide_velocity(r['signed_area'], r['length'], 2), .5)
    aligned, _, _ = align_periodic(p, p+[0, 0, 7.07442], 7.07442)
    np.testing.assert_allclose(aligned, p, atol=1e-12)
    np.testing.assert_allclose(resample(p)[[0, -1]], p[[0, -1]])
    k = np.ones(4, bool)
    first = (k, np.array([1, 1, 0, 0], bool), k, np.ones(4))
    last = (k, np.array([0, 1, 1, 0], bool), k, np.ones(4))
    assert abs(budget(first, last)['balance_error']) < 1e-12
    np.testing.assert_allclose(incremental_reference(np.eye(3), np.eye(3), 2*np.eye(3), [1, 2], [0, 0]), 6*np.eye(3))
    pos = np.array([[215, 200, 0], [215, 200, 1]])
    virial = np.array([np.eye(3), np.eye(3)])
    s = sector_stress(pos, virial, 1, 2)
    np.testing.assert_allclose(s, -np.eye(3)/(3.53721**3/4)*160.21766208)
    field = average_field(pos, virial, 70.7442)
    assert field['count'].sum() == 2
    sample = {'positions_A': pos, 'fault_type': np.array([2, 3]),
              'interatomic_distance_A': np.ones(2)*2.5}
    assert sum(r['atoms'] for r in transition_rows(sample, sample)) == 2
    assert fault_regions(pos, sample['fault_type'], sample['interatomic_distance_A'])['ISF_near_nm2'] > 0
    np.testing.assert_allclose(energy_derivative(p, p, lambda x: polygon_energy(x, lambda t: 1)), 2., atol=1e-10)
    print('Loading, isotropic line energy, ribbon area, periodic alignment, area balance and local stress checks passed.')


if __name__ == '__main__':
    main()
