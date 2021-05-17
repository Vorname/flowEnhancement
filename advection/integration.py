import numpy as np
import sys

sys.path.append('../') # add root code folder to search path for includes

from advection.visualize_flow import *
from advection.reconstruction import *
from advection.global_functions import *
from advection.Log import ProgressBar as progress

def GetRenormalFactor(coeffs ):
    return Energy(coeffs)

def Renormalize(coeffs , e1 ):
    #nb = int(np.sqrt(coeffs.shape[0]))-1
    #c = coeffs.reshape([nb+1, nb+1])
    #c = c[1:,1:]
    #c = c.reshape(nb**2)
    #scale = lambda_vec(nb).reshape([nb+1, nb+1])
    #scale = scale[1:,1:]
    #scale = 1.0/scale.reshape(nb**2)
    #
    #p2 = -2.0*np.sum(c * c * scale * scale)
    #
    #h = p2 + np.sqrt(p2**2 - sum(c*c * scale) + 8.0/(np.pi**2))
    #
    #r = c + 2*h*scale * c
    #
    #res = np.zeros([nb+1, nb+1])
    #res[1:,1:] = r.reshape([nb, nb])
    #res = res.reshape((nb+1)**2)


    # comp = np.abs(GetRenormalFactor(res) - e1)
    # if comp > 10**-7:
    #     print("{}\n".format(comp))

    return coeffs#res

def Euler(coeffs, C, v, time_step, usefft = True):
    c1, _ = Advection(coeffs, C, v, usefft = usefft)   
    return coeffs + c1 * time_step
    

def RungeKutta(coeffs , C , v , time_step , usefft = False):
    e = GetRenormalFactor(coeffs)

    c1_p, c1_n = Advection(coeffs, C, v, usefft = usefft)
    c1_p *= time_step; c1_n *= time_step

    tmp = coeffs + c1_p/2
    c2_p, c2_n = Advection(Renormalize(tmp, e) + c1_n/2, C, v, usefft = usefft)
    c2_p *= time_step; c2_n *= time_step

    tmp = coeffs + c2_p/2
    c3_p, c3_n = Advection(Renormalize(tmp, e) + c2_n/2, C, v, usefft = usefft)
    c3_p *= time_step; c3_n *= time_step

    tmp = coeffs + c3_p
    c4_p, c4_n = Advection(Renormalize(tmp, e) + c3_n, C, v, usefft = usefft)
    c4_p *= time_step; c4_n *= time_step

    a_p = ((1/6)*(c1_p+2*c2_p+2*c3_p+c4_p))
    a_n = ((1/6)*(c1_n+2*c2_n+2*c3_n+c4_n))
    tmp = coeffs+a_p
    return Renormalize(tmp, e) + a_n

def Leapfrog(coeffs , coeffs_lt , C , v , time_step ,  usefft=False):
    #Robert time smoothing constant
    r = 0.02
    coeffs_llt = np.copy(coeffs_lt)

    c_dot_p, c_dot_n = Advection(coeffs, C, v, usefft = usefft)

    temp = np.copy(coeffs)
    coeffs = np.copy(coeffs_lt)
    coeffs_lt = np.copy(temp)

    e = GetRenormalFactor(coeffs)

    coeffs += 2.0 * time_step * c_dot_p
    coeffs = Renormalize(coeffs, e) + 2.0*time_step * c_dot_n

    coeffs_lt = (1 - 2*r) * coeffs_lt + r*(coeffs+coeffs_llt)

    return coeffs, coeffs_lt

def IntegrateSimple(coeffs, viscosity, steps, fps, energy_data=None, coeff_data=None, useEuler = False):
    #print("integrateSimple called with: {}, {}, {}".format(viscosity,steps,fps))
    c = np.copy(coeffs)
    for step in range(0,steps):
        if not energy_data is None:
            energy = Energy(c);
            energy_data[step] = energy
        if not coeff_data is None:
            coeff_data[:,step] = c 
        if useEuler:
            c = Euler(c, None, viscosity, 1/fps, True)
        else:
            c = RungeKutta(c, None, viscosity, 1/fps, True)

def Integrate(coeffs , C , basisfunctions , viscosity , t_max , fps , x1 , x2 , basename,
usefft=False, timeindex = -1, render=True, flow_data = None, print_Progress=True, energy_preserving=False, energy_data=None, coeff_data=None):

    # One could assert a lot of thing here:
    # coeffs fit to basisfunction, shape of parameters
    # type of t_max and fps... 

    n_d = int(np.sqrt(coeffs.shape[0]))

    c = coeffs
    original_energy = Energy(c)
    c_lt = np.copy(coeffs)

    # python 3 supports arbitary intel size, no worries here
    int_t_max = int(t_max * fps)

    for step_count in range(0,int_t_max-1):
        energy = Energy(c);
        if not energy_data is None:
            energy_data[step_count] = energy
        if not coeff_data is None:
            coeff_data[:,step_count] = c 
        if print_Progress:
            progress.ProgressBar(step_count, int_t_max, "Energy_before = {:e} Energy_after = {:e} Energy difference = {:e}".format(original_energy, energy, energy-original_energy), time_index=timeindex)
        if not (flow_data is None):
            flow_data.AddFrameData(c)
        if render:
            flow = Reconstruct_Velocity(c, basisfunctions, usefft)
            visualize_velocity(flow, x1, x2, basename + "{:03}".format(step_count))
        if step_count == 0 or energy_preserving:
            c = RungeKutta(c, C, viscosity, 1/fps, usefft)
        else:
            (c, c_lt) = Leapfrog(c, c_lt, C, viscosity, 1/fps,  usefft)
    if not energy_data is None:
        energy_data[int_t_max-1] = energy
    if not coeff_data is None:
        coeff_data[:,int_t_max-1] = c 
    if not (flow_data is None):
        flow_data.AddFrameData(c)
    if print_Progress:
        progress.ProgressBar(step_count+1, int_t_max, "Energy_before = {:e} Energy_after = {:e} Energy difference = {:e}".format(original_energy, energy, energy-original_energy), time_index=timeindex)

    return
