
import numpy as np
import matplotlib.pyplot as plt

from neutrons_code_L import Capsule


En = np.geomspace(1e-5, 16, 500)[::-1]


params = np.array([[1e30, 0.5, 0.5, 0, 0, 35e-6],
                  [1e32, 0.5, 0.5, 0, 0, 1e-5]])

no_targets = [None, None]

capsule = Capsule(En, params, no_targets)
capsule.solve_neutron_transport()


capsule.animate_spectrum()