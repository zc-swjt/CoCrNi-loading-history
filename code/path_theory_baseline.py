"""Anisotropic plane-strain crack field in the specimen coordinate system."""
import numpy as np

C = {'C11_GPa': 290.73116264133336, 'C12_GPa': 184.239628864,
     'C44_GPa': 144.54858537955556}


def crack_field():
    from ase import units
    from matscipy.fracture_mechanics.crack import CubicCrystalCrack
    return CubicCrystalCrack([0, 1, 0], [0, 0, 1],
                            C11=C['C11_GPa']*units.GPa,
                            C12=C['C12_GPa']*units.GPa,
                            C44=C['C44_GPa']*units.GPa)


def nominal_tensor_modes(points_A, ki, kii):
    from ase import units
    from matscipy.fracture_mechanics.crack import MPa_sqrt_m
    x, y = (np.asarray(points_A)[:, :2]-200).T
    xx, yy, xy = crack_field().stresses(x, y, 0, 0, ki*MPa_sqrt_m, kII=kii*MPa_sqrt_m)
    s = np.zeros((len(x), 3, 3))
    s[:, 0, 0], s[:, 1, 1] = xx, yy
    s[:, 0, 1] = s[:, 1, 0] = xy
    s[:, 2, 2] = C['C12_GPa']/(C['C11_GPa']+C['C12_GPa'])*(xx+yy)
    return s/units.GPa
