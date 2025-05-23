import numpy as np
import pandas as pd

# from neutrons_code import Capsule
# from neutrons_code.functions import neutron_counting

from neutrons_code_L  import Capsule
from neutrons_code_L.functions import neutron_counting

class Moderation_Experiment_DDCRYO:
    def __init__(self, n_runs, time=800):

        self.En = np.geomspace(1e-5, 16, 500)[::-1]
        self.default_params = np.array([[1e30, 0, 1, 0, 0, 35e-6],
                                        [1e32, 0, 1, 0, 0, 1e-5],
                                        [1e32, 0, 0, 0, 1, 1e-5]])

        self.n_runs = n_runs
        self.time = time

        self.H_fraction_hotspot = np.linspace(1e-6,1, n_runs)
        self.fuel_fraction_hotspot = (1 - self.H_fraction_hotspot)

        self.N_hydrogen = 4 * np.pi / 3 * self.default_params[0,5]**3 * self.default_params[0,0] * self.H_fraction_hotspot

        fuel_ions = 4 * np.pi / 3 * ((self.default_params[0,5] + self.default_params[1,5])**3 - self.default_params[0,5]**3) * self.default_params[1,0]
        self.H_fraction_shell = self.N_hydrogen / fuel_ions
        self.fuel_fraction_shell = (1 - self.H_fraction_shell)


        ablator_ions = 4 * np.pi / 3 * ((self.default_params[0,5] + self.default_params[1,5] + self.default_params[2,5])**3 - (self.default_params[0,5] + self.default_params[1,5])**3) * self.default_params[2,0]
        self.H_fraction_ablator = self.N_hydrogen / ablator_ions
        self.carbon_fraction = 1 - self.H_fraction_ablator



        # print(self.H_fraction_hotspot)
        # print(self.H_fraction_shell)
        # print(self.H_fraction_ablator)






    def run_experiment(self):

        self.final_results_hotspot = pd.DataFrame()
        self.final_results_shell = pd.DataFrame()
        self.final_results_ablator = pd.DataFrame()

        for i in range(self.n_runs):
            print(f'run {i+1} of {self.n_runs}')

            no_targets = [None, None, None]


            # need to add an undoped ablator to the first two cases otherwise it looks kinda rubbish


            vary_hotspot_params = np.array([[self.default_params[0,0], 0, self.fuel_fraction_hotspot[i], self.H_fraction_hotspot[i], 0, self.default_params[0,5]],
                                            [self.default_params[1,0], 0, 1, 0, 0, self.default_params[1,5]],
                                            [self.default_params[2,0], 0, 0, 0, 1, self.default_params[2,5]]])

            vary_shell_params = np.array([[self.default_params[0,0], 0, 1, 0, 0, self.default_params[0,5]],
                                          [self.default_params[1,0], 0, self.fuel_fraction_shell[i], self.H_fraction_shell[i], 0, self.default_params[1,5]],
                                          [self.default_params[2,0], 0, 0, 0, 1, self.default_params[2,5]]])

            add_ablator_params = np.array([[self.default_params[0,0], 0, 1, 0, 0, self.default_params[0,5]],
                                           [self.default_params[1,0], 0, 1, 0, 0, self.default_params[1,5]],
                                           [self.default_params[2,0], 0, 0, self.H_fraction_ablator[i], self.carbon_fraction[i], self.default_params[2,5]]])

            # print(vary_hotspot_params)
            # print(vary_shell_params)
            # print(add_ablator_params)

            capsules = [Capsule(self.En, vary_hotspot_params, no_targets),
                        Capsule(self.En, vary_shell_params, no_targets),
                        Capsule(self.En, add_ablator_params, no_targets)]

            [capsule.solve_neutron_transport() for capsule in capsules]


            self.final_results_hotspot = pd.concat((self.final_results_hotspot, neutron_counting(capsules[0], time=self.time)))
            self.final_results_shell = pd.concat((self.final_results_shell, neutron_counting(capsules[1], time=self.time)))
            self.final_results_ablator = pd.concat((self.final_results_ablator, neutron_counting(capsules[2], time=self.time)))

        
        self.final_results_hotspot = self.final_results_hotspot.reset_index().drop(columns='index')
        self.final_results_shell = self.final_results_shell.reset_index().drop(columns='index')
        self.final_results_ablator = self.final_results_ablator.reset_index().drop(columns='index')



