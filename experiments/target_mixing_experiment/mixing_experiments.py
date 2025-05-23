import numpy as np
import matplotlib.pyplot as plt

# from neutrons_code.capsule import Capsule
# from neutrons_code.functions import widths

from neutrons_code_L.capsule import Capsule
from neutrons_code_L.functions import widths


En = np.geomspace(1e-5, 16, 500)[::-1]

n_runs = 10



class Experiment:

    def __init__(self, POORLY_MIXED_RADIUS):

        self.N_TARGET_IONS = 1e15  # poorly and well mixed targets will have the same number of ions

        self.TOTAL_CAPSULE_RADIUS = 500e-6
        self.HOTSPOT_RADIUS = 300e-6  # hotspot may be split into 2 layers at some point, but this is the total size of it
        self.POORLY_MIXED_RADIUS = POORLY_MIXED_RADIUS

        single_hotspot_params = [1e30, 0, 1, 0, 0, self.HOTSPOT_RADIUS]


        # self.r_1 is purely fuel layer outside the hotspot with no target ions mixed into it
        # self.r_2 is fuel layer with well-mixed target 
        # self.r_1 + self.r_2 should be constant, and equal to TOTAL_RADIUS - POORLY_MIXED - HOTSPOT

        self.buffer = 1e-6  # self.buffer is just to ensure that layers dont get too thin for them to be solved
        self.r_2 = np.linspace(self.buffer, self.TOTAL_CAPSULE_RADIUS - self.POORLY_MIXED_RADIUS -self.HOTSPOT_RADIUS - self.buffer, n_runs)
        self.r_1 = self.TOTAL_CAPSULE_RADIUS - self.POORLY_MIXED_RADIUS - self.HOTSPOT_RADIUS - self.r_2


        # p denotes poorly-mixed layer
        # w denotes well-mixed layer

        """
        want well-mixed target to be evenly mixed throughout the regions it is in  (same fraction in each of these layers)
        because all layers have the same number density, calculation is simplified as you only need to find the density in one of the layers
        """


        N_deutrium_p = 4 * np.pi / 3 * ((self.TOTAL_CAPSULE_RADIUS)**3 - (self.TOTAL_CAPSULE_RADIUS - self.POORLY_MIXED_RADIUS)**3) * single_hotspot_params[0]  # all layers have the same number density of 1e30 so its fine to use hotspot params
        self.target_fraction_p = self.N_TARGET_IONS / N_deutrium_p

        N_deuterium_w = 4 * np.pi / 3 * ((self.TOTAL_CAPSULE_RADIUS - self.POORLY_MIXED_RADIUS)**3 - (self.TOTAL_CAPSULE_RADIUS - self.POORLY_MIXED_RADIUS - self.r_2)**3) * single_hotspot_params[0]

        N_wions_inner = self.N_TARGET_IONS / (1 + (N_deutrium_p)/(N_deuterium_w))
        N_wions_outer = self.N_TARGET_IONS - N_wions_inner

        self.target_fraction_w = N_wions_inner / N_deuterium_w
        # target_fraction_w_outer = N_wions_outer / N_deutrium_p


    def run_experiment(self):
        print('starting experiment')
        self.N_captures_p = np.array([])
        self.N_captures_w = np.array([])

        self.total_neutrons = np.array([])

        for i in range(n_runs):
            print(f'run {i+1} of {n_runs}')

            all_params = np.array([[1e30, 0, 1, 0, 0, self.HOTSPOT_RADIUS/2],    # in this version, hotspot split in 2 to test that it gives the same results
                                   [1e30, 0, 1, 0, 0, self.HOTSPOT_RADIUS/2],
                                   [1e30, 0, 1, 0, 0, self.r_1[i]],
                                   [1e30, 0, 1, 0, 0, self.r_2[i]],
                                   [1e30, 0, 1, 0, 0, self.POORLY_MIXED_RADIUS]])

            all_targets = [None, None, None, [['89Y(n,G)', self.target_fraction_w[i]]], [['89Y(n,G)', self.target_fraction_w[i]], ['89Y(n,G)', self.target_fraction_p]]]  # need to remember the ordering of well and poorly mixed targets as they need to be added together


            self.capsule = Capsule(En, all_params, all_targets)

            # for i in range(len(self.capsule.target_info[0])):
            #     self.capsule.target_info[0][i] = np.ones(500) * 1e-29

            self.capsule.solve_neutron_transport()
            # self.total_neutrons = np.append(self.total_neutrons, np.sum(self.capsule.Q[-1]/self.capsule.v[None, :]))
            w = widths(self.capsule.En)
            self.total_neutrons = np.append(self.total_neutrons, np.sum(self.capsule.Q/self.capsule.v*w, axis=(1,2))[-1])

            self.N_captures_p = np.append(self.N_captures_p, np.sum(self.capsule.Q[200, -2:-1]*w)) # target number 3 is currently the poorly-mixed species
            self.N_captures_w = np.append(self.N_captures_w, np.sum(self.capsule.Q[200, -4:-2]*w)) # targets 1 & 2 are currently the well-mixed species



class Experiment_hotspot:
    def __init__(self, POORLY_MIXED_RADIUS):
        self.N_TARGET_IONS = 1e15

        self.TOTAL_CAPSULE_RADIUS = 500e-6
        self.HOTSPOT_RADIUS = 300e-6
        self.POORLY_MIXED_RADIUS = POORLY_MIXED_RADIUS

        self.buffer = 1e-6

        self.r_2 = np.linspace(self.buffer, self.HOTSPOT_RADIUS - self.buffer, n_runs)   # this is the radius of the outer hotspot
        self.r_1 = self.HOTSPOT_RADIUS - self.r_2     # this is the inner hotspot radius


        # calculate poorly mixed fraction
        N_deutrium_p = 4 * np.pi / 3 * (self.TOTAL_CAPSULE_RADIUS**3 - (self.TOTAL_CAPSULE_RADIUS - self.POORLY_MIXED_RADIUS)**3) * 1e30
        self.target_fraction_p = self.N_TARGET_IONS / N_deutrium_p

        # calculate well mixed fraction
        N_deuterium_w = 4 * np.pi / 3 * (self.TOTAL_CAPSULE_RADIUS**3 - self.r_1**3) * 1e30
        self.target_fraction_w = self.N_TARGET_IONS / N_deuterium_w


    def run_experiment(self):
        print('starting experiment')


        self.N_captures_p = np.array([])
        self.N_captures_w = np.array([])

        self.total_neutrons = np.array([])

        for i in range(n_runs):
            print(f'run {i+1} of {n_runs}')

            all_params = np.array([[1e30, 0, 1, 0, 0, self.r_1[i]],
                                   [1e30, 0, 1, 0, 0, self.r_2[i]],
                                   [1e30, 0, 1, 0, 0, self.TOTAL_CAPSULE_RADIUS - self.POORLY_MIXED_RADIUS - self.HOTSPOT_RADIUS],
                                   [1e30, 0, 1, 0, 0, self.POORLY_MIXED_RADIUS]])

            all_targets = [None, [['89Y(n,G)', self.target_fraction_w[i]]], [['89Y(n,G)', self.target_fraction_w[i]]], [['89Y(n,G)', self.target_fraction_w[i]], ['89Y(n,G)', self.target_fraction_p]]]

            self.capsule = Capsule(En, all_params, all_targets)

            # for i in range(len(self.capsule.target_info[0])):
            #     self.capsule.target_info[0][i] = np.ones(500) * 1e-29


            self.capsule.solve_neutron_transport()

            # self.total_neutrons = np.append(self.total_neutrons, np.sum(self.capsule.Q[-1]/self.capsule.v[None, :]))
            w = widths(self.capsule.En)
            self.total_neutrons = np.append(self.total_neutrons, np.sum(self.capsule.Q/self.capsule.v*w, axis=(1,2))[-1])





            self.N_captures_p = np.append(self.N_captures_p, np.sum(self.capsule.Q[200, -1:]*w))
            self.N_captures_w = np.append(self.N_captures_w, np.sum(self.capsule.Q[200, -4:-1]*w))





if __name__ == '__main__':
    experiment = Experiment_hotspot(10e-6)

    experiment.run_experiment()

    capture_ratio = experiment.N_captures_w / experiment.N_captures_p
    # mixing_ratio = (experiment.r_2 + experiment.POORLY_MIXED_RADIUS) / experiment.POORLY_MIXED_RADIUS

    # plt.plot(capture_ratio)
    # plt.show()

    plt.plot(experiment.total_neutrons)
    plt.show()
    

