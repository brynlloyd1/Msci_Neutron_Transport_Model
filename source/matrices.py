import numpy as np
import matplotlib.pyplot as plt

import NeSST as nst

from .functions import init_NeSST, widths


class Matrices:
    
    def __init__(self, En):
        self.En = En
        self.v = np.sqrt(1.6e-19 * En * 1e6 * 2 / 1.67e-27)


    def get_each_matrix(self, kind, params):
        """
        Returns normalised matrix for each kind of scattering

        parameters
        ----------
        kind : str
            type of reaction. Has the form 'elastic/inelastic' + '_' + 'T/D/C' or is just 'hydrogen'
        params : list
            is an element of Capsule.all_params. Contains parameters of a single layer of the capsule.
        """

        if kind == 'elastic_T':
            M = nst.sm.mat_T.elastic_dNdEdmu
            sigma = nst.sm.mat_T.sigma(self.En)
            N = 3
            fraction = params[1]

        elif kind == 'elastic_D':
            M = nst.sm.mat_D.elastic_dNdEdmu
            sigma = nst.sm.mat_D.sigma(self.En)
            N = 2
            fraction = params[2]

        elif kind == 'inelastic_T':
            M = np.transpose(np.trapz(nst.sm.mat_T.n2n_ddx.rgrid,x=nst.sm.mat_T.n2n_mu,axis=1))/2
            sigma = nst.sm.mat_T.n2n_ddx.xsec_interp(self.En)
            fraction = params[1]
            N = 3

        elif kind == 'inelastic_D':
            M = np.transpose(np.trapz(nst.sm.mat_D.n2n_ddx.rgrid,x=nst.sm.mat_D.n2n_mu,axis=1))/2
            sigma = nst.sm.mat_D.n2n_ddx.xsec_interp(self.En)
            fraction = params[2]
            N = 2

        elif kind == 'elastic_H':
            M = nst.sm.mat_H.elastic_dNdEdmu
            sigma = nst.sm.mat_H.sigma(self.En)
            fraction = params[3]
            N = 1

        elif kind == 'elastic_C':
            M =  nst.sm.mat_12C.elastic_dNdEdmu
            sigma = nst.sm.mat_12C.sigma(self.En)
            fraction = params[4]
            N = 12

        elif kind == 'inelastic_C':
            M = nst.sm.mat_12C.inelastic_dNdEdmu
            sigma = nst.sm.mat_12C.sigma(self.En)
            fraction = params[4]
            N = 12

        else:
            raise ValueError ('kind must be elastic/inelastic + _ + D/T/C/H')

        w =  widths(self.En)

        # accounts for scattering below the lowest energy bin in elastic matrices
        if 'inelastic' not in kind:
            for i,E in enumerate(self.En):
                column_integral = np.trapz(M*w[:,None], axis=0)
                alpha = ((N-1)/(N+1))**2
                if alpha * E < self.En[-1]:
                    M[len(self.En)-1,i] += (sigma[i] - column_integral[i])

        # normalisation
        column_integral = np.trapz(M*w[:,None], axis=0)
        column_integral[column_integral == 0] = 1
        M = M / column_integral * sigma

        # approximating integral with sum so need to multiply by bin width
        M *= w[None, :]  # Aidans
        # M *= w[:, None]    # our old one

        # inelastic cross sections are (n,2n) reactions so must create 2 neutrons
        if 'inelastic' in kind:
            M *= 2

        # cross-section goes on diag
        M -= np.diag(sigma)

        # params[0] x fraction is the number density of that species
        M *= 1e-28 * params[0] * fraction * self.v[None, :]

        return M


    def make_total_matrix(self, params, inelastic=True):
        """
        Creates the total scattering matrix for a single layer of the Capsule instance using parameters stored in params

        parameters
        ----------
        kind : str
            type of reaction. Has the form 'elastic/inelastic' + '_' + 'T/D/C' or is just 'hydrogen'
        params : list
            is an element of Capsule.all_params. Contains parameters of a single layer of the capsule.
        """

        M = np.zeros((len(self.En),len(self.En)))


        if params[1] > 0:
            M += self.get_each_matrix('elastic_T', params)

            if inelastic == True:
                M += self.get_each_matrix('inelastic_T', params)
                
        if params[2] > 0:
            M += self.get_each_matrix('elastic_D', params)

            if inelastic == True:
                M += self.get_each_matrix('inelastic_D', params)
        
        # neutron only scatter elastically off hydrogen
        if params[3] > 0:
            M += self.get_each_matrix('elastic_H', params)

        if params[4] > 0:
            M += self.get_each_matrix('elastic_C', params)

            # if inelastic == True:
            #     M += self.get_each_matrix('inelastic_C', params)

        # no scattering out of final energy bin
        M[-1,-1] = 1

        return M    