import numpy as np
import pandas as pd

# from neutrons_code import Capsule
# from neutrons_code.functions import neutron_counting

from neutrons_code_L import Capsule
from neutrons_code_L.functions import neutron_counting

class Moderation_Experiment_DDPDXP:
    def __init__(self, n_runs, time=800):

        self.En = np.geomspace(1e-5, 16, 500)[::-1]


        self.n_runs = n_runs
        self.time = time

        self.H_fraction_hotspot = np.linspace(0,0.35, n_runs)
        self.fuel_fraction_hotspot = (1 - self.H_fraction_hotspot)

        # find number of hydrogen atoms for each of H_fraction_hotspot to then match to hydrogen atoms in ablator

        self.N_hydrogen = 1e30 * self.H_fraction_hotspot * 4 * np.pi/3 * 450e-6**3

        # total capsule size is 0.5mm with 0.05mm thick ablator
        self.H_fraction_ablator = (3 * self.N_hydrogen) / (1e30 * 4 * np.pi * (500e-6**3 - 450e-6**3))
        self.fuel_fraction_ablator = (1 - self.H_fraction_ablator)

    def run_experiment(self):

        self.final_results_hotspot = pd.DataFrame()
        self.final_results_ablator = pd.DataFrame()

        for i in range(self.n_runs):
            print(f'run {i+1} of {self.n_runs}')

            no_targets = [None, None]

            vary_hotspot_params = np.array([[1e30, 0, self.fuel_fraction_hotspot[i], self.H_fraction_hotspot[i], 0, 450e-6],
                                            [1e30, 0, 0, 0, 1, 50e-6]])
            
            vary_ablator_params = np.array([[1e30, 0, 1, 0, 0, 450e-6],
                                            [1e30, 0, 0, self.H_fraction_ablator[i], self.fuel_fraction_ablator[i], 50e-6]])
                                            # [1e30, self.fuel_fraction_ablator[i], self.fuel_fraction_ablator[i], self.H_fraction_ablator[i], 0, 50e-6]])

            capsules = [Capsule(self.En, vary_hotspot_params, no_targets),
                        Capsule(self.En, vary_ablator_params, no_targets)]

            [capsule.solve_neutron_transport() for capsule in capsules]

            self.final_results_hotspot = pd.concat((self.final_results_hotspot, neutron_counting(capsules[0], time=self.time)))
            self.final_results_ablator = pd.concat((self.final_results_ablator, neutron_counting(capsules[1], time=self.time)))

        
        self.final_results_hotspot = self.final_results_hotspot.reset_index().drop(columns='index')
        self.final_results_ablator = self.final_results_ablator.reset_index().drop(columns='index')