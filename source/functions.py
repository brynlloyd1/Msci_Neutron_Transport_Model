import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation
import pandas as pd

import NeSST as nst


def init_NeSST(En):
    """initialises NeSST energy grids using NeSST functions"""

    nst.sm.mat_T.init_energy_grids(En, En)
    nst.sm.mat_D.init_energy_grids(En, En)
    nst.sm.mat_H.init_energy_grids(En, En)
    nst.sm.mat_12C.init_energy_grids(En, En)
    nst.sm.mat_T.init_station_scatter_matrices()
    nst.sm.mat_D.init_station_scatter_matrices()
    nst.sm.mat_H.init_station_scatter_matrices()
    nst.sm.mat_12C.init_station_scatter_matrices()

def widths(En):
    """calculates widths of energy bins"""

    widths = -np.diff(En) 
    widths = np.append(widths,widths[len(En)-2])
    return widths

def gaussian(x, A, mu, sigma, factor):
    num = (x - mu)**2
    den = 2 * sigma**2
    norm = 1/np.sqrt(2*np.pi)/sigma
    return A * np.exp(-num/den) * factor * norm


# =================================================================



def expo(x,a,b,c):
    return a*np.exp(-b*x)+c

def frac_out_coef1(R):
    return expo(R, -7.59384937e+07, 1.25576167e+04, -1.56769298e+06)
def frac_out_coef2(R):
    return expo(R, 10995.26676548, 7864.25972519, 1419.15127435)

def frac_out_coef3(R):
    return expo(R, 4.65006491e-02, 3.88015435e+03, 5.64065298e-01)

def find_frac_out(R,r):
    x = R-r
    a = frac_out_coef1(R)
    b = frac_out_coef2(R)
    c = frac_out_coef3(R)
    return a*x**2 + b*x + c


def L_coef1(R):
    return expo(R, 1.61753029e+08, 1.29059913e+04, 5.07237926e+06)

def L_coef2(R):
    return expo(R, -20837.12529098, 8178.00142305, -4021.13532592)

def L_coef3(R):
    return expo(R, -3.57563588e-01, 3.12312029e+03, 1.94693754e+00)



def find_L(R,r):
    x = R-r
    a = L_coef1(R)
    b = L_coef2(R)
    c = L_coef3(R)

    return (a*x**2 + b*x + c)*x




# =================================================================



def neutron_counting(capsule, time):

    # first index in En for which the energy is less than 1MeV etc.
    POS_1MeV = 98
    POS_100keV = 178
    POS_10keV = 258
    POS_1keV = 339

    # use neutron number to count neutrons, use capsule.Q to find the flux below some energy
    neutron_number = capsule.Q / capsule.v[None, None, :]

    w = widths(capsule.En)

    # :-4 to not count target layers
    # sums from P_energy onwards
    # axis 1 bc specifying the time makes it a 2d array not 3d anymore
    smaller_than_1MeV = np.sum(capsule.Q[time, :-4, POS_1MeV:]*w[POS_1MeV], axis=1)
    smaller_than_100keV = np.sum(capsule.Q[time, :-4, POS_100keV:]*w[POS_100keV], axis=1)
    smaller_than_10keV = np.sum(capsule.Q[time, :-4, POS_10keV:]*w[POS_10keV], axis=1)
    smaller_than_1keV = np.sum(capsule.Q[time, :-4, POS_1keV:]*w[POS_1keV], axis=1)

    results = pd.DataFrame()

    for N,_ in enumerate(capsule.Q[0, :-4]):  # for each layer in capsule+outside

        results[f's1MeV_{N}'] = [smaller_than_1MeV[N]]  # for some reason this first one needs to be a 1-element list or it doesnt work...
        results[f's100keV_{N}'] = smaller_than_100keV[N]
        results[f's10keV_{N}'] = smaller_than_10keV[N]
        results[f's1keV_{N}'] = smaller_than_1keV[N]
    
    return results