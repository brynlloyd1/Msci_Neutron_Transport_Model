import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

import time

import NeSST as nst

from .matrices import Matrices
from .targets import Targets
from .source import get_source_DT, get_source_D
from .ODE import dQdt
from .functions import gaussian, init_NeSST, widths


"""
all_params =
0 = n
1 = T_frac
2 = D_frac
3 = H_frac
4 = C_frac
5 = L

all_targets = 
[[['171Tm(n,G)', 0.1],['reaction name', fraction]], None, ...for each layer] => 2 targets in hotspot and none in shell
reaction names are in the dictionary in init
if a layer has no targets, use None
only up to 2 targets total
"""


class Capsule(Matrices, Targets):

    # variable that will be set as True once the first instance of Capsule is created
    initialised = False

    def __init__(self, En, all_params, all_targets, inelastic=True, method='RK23'):
        """
        Parameters
        ----------
        En : numpy array
            array of (decreasing) energy. Is a class attribute because it should be the same for all instances of Capsule...
        all_params : numpy array
            array where each element is a list of the parameters of each layer in the capsule
        all_targets : list
            list containing target position, reaction, concentration
        inelastic : Boolean
            determines whether inelastic scattering matrices are used (default is True)
        method : str
            method to be used by scipy.integrate.solve_ivp (default is 'RK23' but 'BDF' is also used)
        """

        if Capsule.initialised == False:
            # all things done inside this if statement will only be run on the creation of the first instance

            Capsule.En = En
            Capsule.v = np.sqrt(1.6e-19 * En * 1e6 * 2 / 1.67e-27)
            Capsule.t = np.linspace(0, 2.5e-10, 1000)
            
            print('initialising NeSST')
            init_NeSST(En)

            Capsule.initialised = True

        self.all_params = all_params

        # create a single scattering matrix for each layer in the capsule
        self.matrices = [self.make_total_matrix(params, inelastic=inelastic) for params in all_params]

        # create targets
        Targets.__init__(self, En, all_targets)
        self.target_info = self.get_target_info()  # target info [[sigmas], [fractions], [positions]]
        # add target cross-sections to the scattering matrices so that captured neutrons are removed from the system
        for i in range(len(self.target_info[0])):
            try:
                self.matrices[self.target_info[2][i]] -= np.diag(self.target_info[0][i]) * self.v * self.all_params[self.target_info[2][i], 0] * self.target_info[1][i] * 1e-28
            except:
                raise IndexError ('Tried to add a target to a layer that is outside the capsule')

        # source changes if there is no tritium in capsule
        if self.all_params[0,1] > 0:
            self.S = get_source_DT(self.En, self.all_params[0])
        else:
            self.S = get_source_D(self.En)

        # method for solve_ivp
        self.method = method


    def solve_neutron_transport(self):
        """
        uses scipy.integrate.solve_ivp to return solution to dQdt as a [len(t), 500 * (#layers + extras...)] array
        currently has 4 additional layers for target ion species
        """

        Q0 = np.zeros(len(self.En) * (len(self.all_params) + 5))  # array of length (500 * number of layers). Number of layers is capsule layers +1 for outside and +4 for target 'collectors'
        start = time.time()
        Q = solve_ivp(dQdt, t_span=(0,max(self.t)), y0=Q0, t_eval=self.t, args=(self.En, self.v, self.matrices, self.all_params, self.target_info, self.S), method=self.method)
        print(f'time taken to solve ODEs: {time.time()-start}')

        self.Q = np.reshape(np.transpose(Q.y), (len(self.t), len(self.all_params)+5, len(self.En)))
        

    def animate_spectrum(self):
        """animates neutron spectrum (Capsule.Q[:, :-2000])"""

        def animate(f):
            ax.clear()
            ax.set_title(f'time: {self.t[f]*1e12:.0f} ps')
            for N in range(len(self.Q[0])-4):
                ax.loglog(self.En, self.Q[f, N], '-', label=f'layer {N}')
            ax.set_ylim((np.max(self.Q)*1e-15, 10*np.max(self.Q)))
            ax.set_xlabel('Energy [MeV]')
            ax.set_ylabel('neutron flux [A.U]')
        fig,ax = plt.subplots(figsize=(10,10))
        anim = FuncAnimation(fig, animate, frames=len(self.t), interval=1)
        plt.show()



    def animate_target_spectra(self):
        """animates the spectrum of neutrons 'captured' by target species (Capsule.Q[:, -2000:])"""

        def animate_target(f):
            ax.clear()
            ax.set_title(f'time: {self.t[f]*1e12:.0f} ps')
            for i,Q in enumerate(self.Q[f, -4:]):
                plt.loglog(self.En, Q, label=f'target {i+1}: {np.sum(Q)*1e9:.2f}')
            ax.set_ylim((np.max(self.Q[:,-4:])*1e-15, np.max(self.Q[:,-4:]*10)))
            ax.legend(title=r'$10^{-9}$ neutrons captured')


        fig,ax = plt.subplots(figsize=(10,10))
        anim=FuncAnimation(fig, animate_target, frames=len(self.t), interval=1)
        plt.show()