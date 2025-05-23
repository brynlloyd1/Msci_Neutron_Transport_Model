import numpy as np
import NeSST as nst

########################
# Primary reactivities #
########################

 
# Bosh Hale DT and DD reactivities
# Taken from Atzeni & Meyer ter Vehn page 19

# Output in m3/s, Ti in keV

def reac_DT(Ti):
    C1 = 643.41e-22
    xi = 6.6610*Ti**(-0.333333333)
    eta = 1-np.polyval([-0.10675e-3,4.6064e-3,15.136e-3,0.0e0],Ti)/np.polyval([0.01366e-3,13.5e-3,75.189e-3,1.0e0],Ti)

    return C1*eta**(-0.833333333)*xi**2*np.exp(-3*eta**(0.333333333)*xi)

 

def reac_DD(Ti):
    C1 = 3.5741e-22
    xi = 6.2696*Ti**(-0.333333333)
    eta = 1-np.polyval([5.8577e-3,0.0e0],Ti)/np.polyval([-0.002964e-3,7.6822e-3,1.0e0],Ti)

    return C1*eta**(-0.833333333)*xi**2*np.exp(-3*eta**(0.333333333)*xi)

 

# Gaussian "Brysk"

def Qb(Ein,mean,variance):
    spec = np.exp(-(Ein-mean)**2/2.0/variance)/np.sqrt(2*np.pi*variance)

    return spec

 

def yield_from_dt_yield_ratio(reaction,dt_yield,Ti,frac_D=0.5,frac_T=0.5):

    ''' Reactivity ratio to predict yield from the DT yield assuming same volume and burn time
        rate_ij = (f_{i}*f_{j}*sigmav_{i,j}(T))/(1+delta_{i,j})  # dN/dVdt
        yield_ij = (rate_ij/rate_dt)*yield_dt
 
        Note that the TT reaction produces two neutrons.
    '''
    
    if reaction == 'dd':
        ratio = (0.5*frac_D*reac_DD(Ti))/(frac_T*reac_DT(Ti))

    return ratio*dt_yield

 

###############################################################################

# Ballabio fits, see Table III of L. Ballabio et al 1998 Nucl. Fusion 38 1723 #

###############################################################################

 

# Returns the mean and variance based on Ballabio

# Tion in keV

def DTprimspecmoments(Tion):
    # Mean calculation
    a1 = 5.30509
    a2 = 2.4736e-3
    a3 = 1.84
    a4 = 1.3818

    mean_shift = a1*Tion**(0.6666666666)/(1.0+a2*Tion**a3)+a4*Tion

    # keV to MeV

    mean_shift /= 1e3
    mean = 14.021 + mean_shift

 
    # Variance calculation

    omega0 = 177.259
    a1 = 5.1068e-4
    a2 = 7.6223e-3
    a3 = 1.78
    a4 = 8.7691e-5

    delta = a1*Tion**(0.6666666666)/(1.0+a2*Tion**a3)+a4*Tion

 

    C = omega0*(1+delta)
    FWHM2    = C**2*Tion
    variance = FWHM2/(2.35482)**2

    # keV^2 to MeV^2
    variance /= 1e6

    return mean, variance

 

# Returns the mean and variance based on Ballabio

# Tion in keV

def DDprimspecmoments(Tion):
    # Mean calculation
    a1 = 4.69515
    a2 = -0.040729
    a3 = 0.47
    a4 = 0.81844

    mean_shift = a1*Tion**(0.6666666666)/(1.0+a2*Tion**a3)+a4*Tion

    # keV to MeV

    mean_shift /= 1e3
    mean = 2.4495 + mean_shift

    # Variance calculation

    omega0 = 82.542
    a1 = 1.7013e-3
    a2 = 0.16888
    a3 = 0.49
    a4 = 7.9460e-4

    delta = a1*Tion**(0.6666666666)/(1.0+a2*Tion**a3)+a4*Tion

    C = omega0*(1+delta)
    FWHM2    = C**2*Tion
    variance = FWHM2/(2.35482)**2
    # keV^2 to MeV^2
    variance /= 1e6
    return mean, variance


def get_source_DT(En, params):
    Tion = 10.0 # keV

    DTmean,DTvar = DTprimspecmoments(Tion)
    DDmean,DDvar = DDprimspecmoments(Tion)

    Y_DD = yield_from_dt_yield_ratio('dd',1.0,Tion)

    # fractions should sum to 1?
    frac_T = params[1]/(params[1] + params[2])
    frac_D = params[2]/(params[1] + params[2])
    Y_TT = nst.yield_from_dt_yield_ratio('tt', 1.0, Tion, frac_D=frac_T, frac_T=frac_T)

    dNdE_DT = Qb(En, DTmean, DTvar)
    dNdE_DD = Y_DD*Qb(En, DDmean, DDvar)
    dNdE_TT = Y_TT*nst.dNdE_TT(En, Tion)


    return dNdE_DT + dNdE_DD + dNdE_TT[::-1]



def get_source_D(En):
    Tion = 10.0 # keV

    # DTmean,DTvar = DTprimspecmoments(Tion)
    DDmean,DDvar = DDprimspecmoments(Tion)

    Y_DD = 1.0

    # dNdE_DT = Qb(En,DTmean,DTvar)
    dNdE_DD = Y_DD*Qb(En,DDmean,DDvar)


    return dNdE_DD