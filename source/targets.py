import numpy as np

import os


class Targets:

    def __init__(self, En, all_targets):
        self.En = En
        self.all_targets = all_targets

        Targets.target_dict = {'89Y(n,G)': str(os.getcwd()) + '/neutrons_code/target_cross_sections/89Y_n,g_ENDF-B_VIII.csv',
                                '89Y(n,2n)': str(os.getcwd()) + '/neutrons_code/target_cross_sections/89Y_n,2n_ENDF-B_VIII.csv',
                                '169Tm(n,2n)': str(os.getcwd()) + '/neutrons_code/target_cross_sections/169Tm_n,2n_ENDF-B_VIII.csv',
                                '171Tm(n,G)': str(os.getcwd()) + '/neutrons_code/target_cross_sections/171Tm_n,g_JENDL-5.csv'}


    def get_target_info(self):

        # create the case of no targets, then append for each target up to 4 targets
        target_sigma_list = np.zeros((4, len(self.En)))
        target_fraction_list = np.zeros(4)
        target_positions = np.zeros(4, dtype=int)  # must be integers as these are used as indices

        target_counter = 0

        for layer_number, layer in enumerate(self.all_targets):
            if layer is None:
                continue

            for target in layer:
                target_counter += 1  # dont try and replace with an enumerate bc it resets with each layer

                if target_counter > 4:
                    raise ValueError('Currently only works for 4 targets. Would need to add additional layer to ODE for each additional target')
                
                reaction = target[0]
                fraction = target[1]

                data_x, data_y, _ = np.genfromtxt(self.target_dict[reaction], dtype=None, unpack= True, encoding=None)
                data_x /= 1e6    # convert from eV to MeV
                reaction_sigma = np.interp(self.En[::-1], data_x, data_y)[::-1]
                reaction_sigma[-1] = 1  # to prevent scattering below the lowest bin

                target_sigma_list[target_counter -1] = reaction_sigma
                target_fraction_list[target_counter -1] = fraction
                target_positions[target_counter -1] = layer_number

        return [target_sigma_list, target_fraction_list, target_positions]