import numpy as np
import matplotlib.pyplot as plt

# from neutrons_code.capsule import Capsule
# from neutrons_code.functions import widths

from neutrons_code_L.capsule import Capsule
from neutrons_code_L.functions import widths

En = np.geomspace(1e-5, 16, 500)[::-1]


class Experiment:

    def __init__(self, ablator_thickness, time, n_runs=10):

        self.N_TARGET_IONS = 1e15  # poorly and well mixed targets will have the same number of ions
        self.TOTAL_CAPSULE_RADIUS = 500e-6

        self.FUEL_DENSITY = 1e30
        self.ABLATOR_DENSITY = 1e30


        self.ablator_thickness = ablator_thickness
        self.buffer = 5e-6


        self.time = time
        self.n_runs = n_runs

        self.outer_hotspot = np.linspace(self.buffer, self.TOTAL_CAPSULE_RADIUS - self.ablator_thickness - self.buffer, n_runs)
        self.inner_hotspot = self.TOTAL_CAPSULE_RADIUS - self.ablator_thickness - self.outer_hotspot


        # volume_inner = 4 * np.pi / 3 * self.inner_hotspot**3
        volume_outer = 4 * np.pi / 3 * ((self.inner_hotspot + self.outer_hotspot)**3 - self.inner_hotspot**3)
        volume_ablator = 4 * np.pi / 3 * ((self.TOTAL_CAPSULE_RADIUS)**3 - (self.TOTAL_CAPSULE_RADIUS - self.ablator_thickness)**3)
        
        n_p = self.N_TARGET_IONS / volume_ablator
        self.target_fraction_p = n_p / self.ABLATOR_DENSITY

        n_w = self.N_TARGET_IONS / (volume_outer + volume_ablator)
        self.target_fraction_w_outer = n_w / self.FUEL_DENSITY
        self.target_fraction_w_ablator = n_w / self.ABLATOR_DENSITY







    def run_experiment(self):

        self.N_captures_p = np.array([])
        self.N_captures_w = np.array([])


        self.N_captures_w_outer = np.array([])
        self.N_captures_w_ablator = np.array([])

        self.total_neutrons = np.array([])


        for i in range(self.n_runs):
            print(f'run {i+1} of {self.n_runs}')

            all_params = np.array([[self.FUEL_DENSITY, 0, 1, 0, 0, self.inner_hotspot[i]],
                                    [self.FUEL_DENSITY, 0, 1, 0, 0, self.outer_hotspot[i]],
                                    [self.ABLATOR_DENSITY, 0, 0, 0, 1, self.ablator_thickness]])

            all_targets = [None, [['171Tm(n,G)', self.target_fraction_w_outer[i]]], [['171Tm(n,G)', self.target_fraction_w_ablator[i]], ['171Tm(n,G)', self.target_fraction_p]]]

            self.capsule = Capsule(En, all_params, all_targets, method='RK23', inelastic=True)

            # replace cross-sections with flat cross-sections
            # for i in range(len(self.capsule.target_info[0])):
            #     self.capsule.target_info[0][i] = np.ones(500) * 1e-20

            self.capsule.solve_neutron_transport()
            # self.capsule.animate_spectrum()
            # self.capsule.animate_target_spectra()
            w = widths(self.capsule.En)
            self.total_neutrons = np.append(self.total_neutrons, np.sum(self.capsule.Q/self.capsule.v*w, axis=(1,2))[-1])


            self.N_captures_p = np.append(self.N_captures_p, np.sum(self.capsule.Q[self.time, -2]*w))
            self.N_captures_w = np.append(self.N_captures_w, np.sum(self.capsule.Q[self.time, -4:-2]*w))

            self.N_captures_w_outer = np.append(self.N_captures_w_outer, np.sum(self.capsule.Q[self.time, -4]*w))
            self.N_captures_w_ablator = np.append(self.N_captures_w_ablator, np.sum(self.capsule.Q[self.time, -3]*w))




e = Experiment(10e-6, 500, n_runs=3)
# print(e.target_fraction_p, e.target_fraction_w)
# e.run_experiment()



# print(e.ablator_thickness)





