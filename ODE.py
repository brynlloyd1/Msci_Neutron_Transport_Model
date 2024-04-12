import numpy as np

from .functions import *


def dQdt(t, Q, En, v, matrices, all_params, target_info, S):
    """function to be used by solve_ivp in Capsule.solve_neutron_transport"""
    # L = all_params[:,5]  # make L = 0.75 * radius??

    # currently, 90% of neutrons that leave a capsule due to the -v/L term are put into the next layer outward. 10% go 'inwards' a layer
    # frac_out = 0.9

    # layer thicknesses are given in params
    layer_thickness = all_params[:,5]
    # R is the radius of the outer edge of each layer
    R = [np.sum(layer_thickness[:i+1], axis=0) for i,_ in enumerate(layer_thickness)]

    L = [0.75*R[0]]
    frac_out = [1]

    R_inner = R[0]
    for i in range(1, len(R)):
        L.append(find_L(R[i], R_inner))
        frac_out.append(find_frac_out(R[i], R_inner))
        R_inner = R[i]




    target_sigma_list = target_info[0]
    target_fraction_list = target_info[1]
    target_positions = target_info[2]



    # if you want to partition the hotspot to allow for targets to be mixed in part of the fusioning region, that needs to be changed manually

    Q = np.reshape(Q, (len(all_params)+5, len(En)))
    # print(Q)

    # dQdt_total = np.zeros(np.shape(Q))
    dQdt_total = np.array([])

    # for N in range(len(Q)):

    #     # =====================================================================================

    #     # HOTSPOT

    #     if N == 0:
    #         """ hotspot """

    #         # print('hotspot!!!')

    #         source_normalisation_factor = 100 #634521429363.2875 * 100/5.20682964e9

    #         gaussian_S = gaussian(t, S, 75e-12, 25e-12, source_normalisation_factor) * (1 - all_params[0,3] - all_params[0,4])  # source is scaled by the amount of fuel in the hotspot
    #         dQdt = np.dot(matrices[N],Q[N]) + v*gaussian_S - Q[N]*v/L[N] + (1-frac_out)*Q[N+1]*v/L[N+1]
    #         """(differential + total + any target) cross-section * neutrons in layer + Source*v  - neutrons lost due to v/L term  + neutrons lost from outer layer (radially in  (1-frac_out))"""
    #     # =====================================================================================

    #     # SHELL / ABLATOR

    #     elif N==1 and len(Q) == 7:
    #         """ shell in a 2-layer capsule ( special case) """

    #         dQdt = np.dot(matrices[N],Q[N]) - Q[N]*v/L[N] + Q[N-1]*v/L[N-1]
    #         """(differential + total + any target) cross-section * neutrons in layer  - neutrons lost due to v/L term + neutrons lost from hotspot(all radially outwards so no frac_out)"""
       
    #     elif  N==1:
    #         """ first layer thats not hotspot """

    #         dQdt = np.dot(matrices[N],Q[N]) - Q[N]*v/L[N] + Q[N-1]*v/L[N-1] + (1-frac_out)*Q[N+1]*v/L[N+1]
    #         """(differential + total + any target) cross-section * neutrons in layer   - neutrons lost due to v/L term + neutrons lost from hotspot(all radially outwards so no frac_out) + neutrons lost from outer layer ( 1-frac_out for radially inwards)"""
       

    #     elif N == len(Q)-6:
    #         """ last layer before outside """

    #         dQdt = np.dot(matrices[N],Q[N]) - Q[N]*v/L[N] + frac_out*Q[N-1]*v/L[N-1]
    #         """(differential + total + any target) cross-section * neutrons in layer   - neutrons lost due to v/L term + neutrons lost outwards from inner layer (frac_out)"""
       
    #     # =====================================================================================

    #     # OUTSIDE

    #     elif N == len(Q)-5:
    #         """ outside """

    #         dQdt = frac_out*Q[N-1]*v/L[N-1]
    #         """frac_out of neutrons lost due to v/L term of last shell"""
    #     # =====================================================================================

    #     # TARGETS (these are collector layers to analyse the spectrum of the neutrons that interact with the targets)
    #     # ODE in these layers solves for neutron number not neutron flux so there is a factor of v missing from these lines
       
    #     elif N == len(Q)-4:
    #         """ target 1 """

    #         dQdt = np.dot(np.diag(target_sigma_list[0]),Q[target_positions[0]]) * all_params[target_positions[0],0] * 1e-28 * target_fraction_list[0]
    #         """number of neutrons in the shell where the target is located * sigma of target * v"""

    #     elif N == len(Q)-3:
    #         """ target 2 """

    #         dQdt = np.dot(np.diag(target_sigma_list[1]),Q[target_positions[1]]) * all_params[target_positions[1],0] * 1e-28 * target_fraction_list[1]
    #         """number of neutrons in the shell where the target is located * sigma of target * v"""

    #     elif N == len(Q)-2:
    #         """ target 3 """

    #         dQdt = np.dot(np.diag(target_sigma_list[2]),Q[target_positions[2]]) * all_params[target_positions[2],0] * 1e-28 * target_fraction_list[2]
    #         """number of neutrons in the shell where the target is located * sigma of target * v"""

    #     elif N == len(Q)-1:
    #         """ target 4 """

    #         dQdt = np.dot(np.diag(target_sigma_list[3]),Q[target_positions[3]]) * all_params[target_positions[3],0] * 1e-28 * target_fraction_list[3]
    #         """number of neutrons in the shell where the target is located * sigma of target * v"""
    #     # =====================================================================================

    #     # SHELL again

    #     else:
    #         """ general non-fusioning layer inside capsule """

    #         dQdt = np.dot(matrices[N],Q[N]) - Q[N]*v/L[N] + frac_out*Q[N-1]*v/L[N-1] + (1-frac_out)*Q[N+1]*v/L[N+1]
    #         """(differential + total + any target) cross-section * neutrons in layer  - neutrons lost due to v/L term + neutrons lost outwards from inner layer (frac_out) + neutrons lost from outer layer (1-frac_out for radially inwards)"""

    #     # =====================================================================================


    #     dQdt_total = np.append(dQdt_total, dQdt)

    # return dQdt_total.flatten()


    for N in range(len(Q)):

        # =====================================================================================

        # HOTSPOT

        if N == 0:
            """ hotspot """

            # print('hotspot!!!')

            source_normalisation_factor = 100 #634521429363.2875 * 100/5.20682964e9

            gaussian_S = gaussian(t, S, 75e-12, 25e-12, source_normalisation_factor) * (1 - all_params[0,3] - all_params[0,4]) * R[0]**3/R[1]**3  # source is scaled by the amount of fuel in the hotspot
            dQdt = np.dot(matrices[N],Q[N]) + v*gaussian_S - Q[N]*v/L[N] + (1-frac_out[N+1])*Q[N+1]*v/L[N+1]
            """(differential + total + any target) cross-section * neutrons in layer + Source*v  - neutrons lost due to v/L term  + neutrons lost from outer layer (radially in  (1-frac_out))"""
        # =====================================================================================

        # SHELL / ABLATOR

        elif N==1 and len(Q) == 7:
            """ shell in a 2-layer capsule ( special case) """
            gaussian_S = gaussian(t, S, 75e-12, 25e-12, source_normalisation_factor) * (1 - all_params[0,3] - all_params[0,4]) * (R[1]**3-R[0]**3)/R[1]**3 
            dQdt = np.dot(matrices[N],Q[N]) - Q[N]*v/L[N] + Q[N-1]*v/L[N-1]+ gaussian_S*v
            """(differential + total + any target) cross-section * neutrons in layer  - neutrons lost due to v/L term + neutrons lost from hotspot(all radially outwards so no frac_out)"""
       
        elif  N==1:
            """ first layer thats not hotspot """
            gaussian_S = gaussian(t, S, 75e-12, 25e-12, source_normalisation_factor) * (1 - all_params[0,3] - all_params[0,4]) * (R[1]**3-R[0]**3)/R[1]**3
            dQdt = np.dot(matrices[N],Q[N]) - Q[N]*v/L[N] + Q[N-1]*v/L[N-1] + (1-frac_out[N+1])*Q[N+1]*v/L[N+1] + gaussian_S*v
            """(differential + total + any target) cross-section * neutrons in layer   - neutrons lost due to v/L term + neutrons lost from hotspot(all radially outwards so no frac_out) + neutrons lost from outer layer ( 1-frac_out for radially inwards)"""
       

        elif N == len(Q)-6:
            """ last layer before outside """

            dQdt = np.dot(matrices[N],Q[N]) - Q[N]*v/L[N] + frac_out[N-1]*Q[N-1]*v/L[N-1]
            """(differential + total + any target) cross-section * neutrons in layer   - neutrons lost due to v/L term + neutrons lost outwards from inner layer (frac_out)"""
       
        # =====================================================================================

        # OUTSIDE

        elif N == len(Q)-5:
            """ outside """

            dQdt = frac_out[N-1]*Q[N-1]*v/L[N-1]
            """frac_out of neutrons lost due to v/L term of last shell"""
        # =====================================================================================

        # TARGETS (these are collector layers to analyse the spectrum of the neutrons that interact with the targets)
        # ODE in these layers solves for neutron number not neutron flux so there is a factor of v missing from these lines
       
        elif N == len(Q)-4:
            """ target 1 """

            dQdt = np.dot(np.diag(target_sigma_list[0]),Q[target_positions[0]]) * all_params[target_positions[0],0] * 1e-28 * target_fraction_list[0]
            """number of neutrons in the shell where the target is located * sigma of target * v"""

        elif N == len(Q)-3:
            """ target 2 """

            dQdt = np.dot(np.diag(target_sigma_list[1]),Q[target_positions[1]]) * all_params[target_positions[1],0] * 1e-28 * target_fraction_list[1]
            """number of neutrons in the shell where the target is located * sigma of target * v"""

        elif N == len(Q)-2:
            """ target 3 """

            dQdt = np.dot(np.diag(target_sigma_list[2]),Q[target_positions[2]]) * all_params[target_positions[2],0] * 1e-28 * target_fraction_list[2]
            """number of neutrons in the shell where the target is located * sigma of target * v"""

        elif N == len(Q)-1:
            """ target 4 """

            dQdt = np.dot(np.diag(target_sigma_list[3]),Q[target_positions[3]]) * all_params[target_positions[3],0] * 1e-28 * target_fraction_list[3]
            """number of neutrons in the shell where the target is located * sigma of target * v"""
        # =====================================================================================

        # SHELL again

        else:
            """ general non-fusioning layer inside capsule """

            dQdt = np.dot(matrices[N],Q[N]) - Q[N]*v/L[N] + frac_out[N-1]*Q[N-1]*v/L[N-1] + (1-frac_out[N+1])*Q[N+1]*v/L[N+1]
            """(differential + total + any target) cross-section * neutrons in layer  - neutrons lost due to v/L term + neutrons lost outwards from inner layer (frac_out) + neutrons lost from outer layer (1-frac_out for radially inwards)"""

        # =====================================================================================


        dQdt_total = np.append(dQdt_total, dQdt)

    return dQdt_total.flatten()