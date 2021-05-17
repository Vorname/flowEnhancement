import numpy as np
import sys

sys.path.append('../') # add root code folder to search path for includes

from advection.reconstruction import *
from advection.projection import *
from advection.Log import Logger
from advection.Log import ProgressBar as prog
from advection.advection_matrix import AdvectionMatrix

def Basisfunction_Velocity(k1 , k2 , x, y):
    if(k1==0 or k2==0):
        return np.array([0,0]);
    return 1/(k1*k1+k2*k2)*np.array([k2*np.sin(k1*x)*np.cos(k2*y),-k1*np.cos(k1*x)*np.sin(k2*y)]);

def Basisfunction_Vorticity(k1 , k2 , x, y):
    return np.sin(k1*x)*np.sin(k2*y);

def lambda_k(k1 , k2 ):
        return -(k1**2 + k2**2);

def lambda_vec(nb ):
    res = np.zeros([nb+1, nb+1])
    res[1:,1:] = np.array([[lambda_k(k1,k2) for k1 in range(1,nb+1)] for k2 in range(1,nb+1)])
    return res.reshape((nb+1)**2)

def IsIndex(i1, i2, number_basis_functions):
    return 0 < i1 and i1 <= number_basis_functions and 0 < i2 and i2 <= number_basis_functions;

def Index(i1, i2, number_basis_functions):
    return i1*(number_basis_functions+1)+i2;

def Energy(coeffs ):
    nb = int(np.sqrt(coeffs.shape[0]))
    scale = np.zeros([nb, nb])
    scale[1:,1:] = np.array([[1/-lambda_k(k1,k2) for k1 in range(1,nb)] for k2 in range(1,nb)])
    scale = scale.reshape((nb)**2)
    c = coeffs * coeffs * scale
    return (np.pi**2/8) * np.sum(c)

def BuildAdvectionMatrix(number_basis_functions , usefft = False, printProgress=False, time_index=-1, useFullShape=False):
    n_b_squared = (number_basis_functions+1)**2;
    if useFullShape:
        res = np.zeros((number_basis_functions+1,number_basis_functions+1,number_basis_functions+1,number_basis_functions+1,number_basis_functions+1,number_basis_functions+1))
        usefft = False
    else:
        res = AdvectionMatrix([n_b_squared, n_b_squared, n_b_squared]);
    if printProgress:
        print()
    if usefft:
        max_steps=0
        current_step=0
        if printProgress:
            max_steps = number_basis_functions**2
            prog.ProgressBar(current_step, max_steps, "", time_index=time_index);
            current_step += 1
        coeffs_flow = np.zeros([number_basis_functions+1, number_basis_functions+1]);
        coeffs_vorticity = np.zeros([number_basis_functions+1, number_basis_functions+1]);

        helper = np.zeros([2, n_b_squared, n_b_squared]);
        for i1 in range(1, number_basis_functions+1):
            for i2 in range(1, number_basis_functions+1):
                helper[0,i1*(number_basis_functions+1)+i2,i1*(number_basis_functions+1)+i2] = -i2**2/(i1**2 + i2**2)
                helper[1,i1*(number_basis_functions+1)+i2,i1*(number_basis_functions+1)+i2] = i1**2/(i1**2 + i2**2);

        for i1 in range(1,number_basis_functions+1):
            for i2 in range(1,number_basis_functions+1):
                if printProgress:
                    prog.ProgressBar(current_step, max_steps, "", time_index=time_index);
                    current_step += 1
                coeffs_vorticity[i1,i2]=1;
                coeffs_vorticity=coeffs_vorticity.reshape(n_b_squared)
                for j1 in range(1,number_basis_functions+1):
                    for j2 in range(1,number_basis_functions+1):
                        coeffs_flow[j1,j2]=1;
                        coeffs_flow=coeffs_flow.reshape(n_b_squared)
                        vorticity=Reconstruct_Vorticity(coeffs_vorticity, n_b_squared, n_b_squared, res_modifier=4);
                        velocity=Reconstruct_Velocity(coeffs_flow, None, usefft=True, res_modifier=4, res=[n_b_squared,n_b_squared]);

                        x = -velocity[1] * vorticity
                        y = velocity[0] * vorticity

                        x_proj, y_proj = Project_Velocity(x, y, number_basis_functions)

                        advx = np.dot(helper[0], x_proj.reshape([n_b_squared]))
                        advy = np.dot(helper[1], y_proj.reshape([n_b_squared]))

                        res.Set([i1,i2,j1,j2], (-advx+advy).reshape([number_basis_functions+1,number_basis_functions+1]));

                        coeffs_flow=coeffs_flow.reshape([number_basis_functions+1,number_basis_functions+1])
                        coeffs_flow[j1,j2]=0;
                coeffs_vorticity=coeffs_vorticity.reshape([number_basis_functions+1,number_basis_functions+1])
                coeffs_vorticity[i1,i2]=0;
        if printProgress:
            prog.ProgressBar(current_step, max_steps, "", time_index=time_index);
            current_step += 1
    else:
        n_b = number_basis_functions+1;
        max_steps=(n_b-1)**2;
        current_step=0;
        #this code ported from Mathematica DON'T TOUCH UNDER ANY CIRCUMSTANCES
        for k1 in range(1,n_b):
            for k2 in range(1,n_b):
                if printProgress:
                    prog.ProgressBar(current_step, max_steps, "", time_index=time_index);
                    current_step += 1
                for i1 in range(1,n_b):
                    for i2 in range(1, n_b):
                        scale = 1/(-4*(i1**2+i2**2));
                        for j1 in range(1, n_b):
                            for j2 in range(1,n_b):
                                if k1 == i1 + j1 and k2 == i2 + j2:
                                    res[k1, k2, j1, j2, i1, i2] += scale*(i1*j2 - i2*j1);
                                if k1 == -i1 + j1 and k2 == -i2 + j2:
                                    res[k1, k2, j1, j2, i1, i2] += scale*((-i1)*j2 - (-i2)*j1);
                                if k1 == -i1 + j1 and k2 == i2 - (-j2):
                                    res[k1, k2, j1, j2, i1, i2] += scale*((-i1)*(-j2) + i2*j1);
                                if k1 == i1 + j1 and k2 == (-i2)-(-j2):
                                    res[k1, k2, j1, j2, i1, i2] += scale*(i1*(-j2)+(-i2)*j1);
                                if k1 == i1 + j1 and k2 == i2 - j2:
                                    res[k1, k2, j1, j2, i1, i2] += scale*(i1*j2 + i2*j1);
                                if k1 == i1 - j1 and k2 == i2 + j2:
                                    res[k1, k2, j1, j2, i1, i2] += -scale*(i1*j2 + i2*j1);
                                if k1 == i1 - j1 and k2 == -i2 - (-j2):
                                    res[k1, k2, j1, j2, i1, i2] += -scale*(i1*(-j2)-(-i2)*j1);
                                if k1 == i1 - j1 and k2 == i2 - j2:
                                    res[k1, k2, j1, j2, i1, i2] += -scale*(i1*j2 - i2*j1);
                                if k1 == (-i1)-(-j1) and k2 == i2 - j2:
                                    res[k1, k2, j1, j2, i1, i2] += -scale*(-i1*j2 - i2*(-j1));
        if printProgress:
            prog.ProgressBar(current_step, max_steps, "", time_index=time_index);
            current_step += 1

    return res;

def Advection(w , C , viscosity , f = None, usefft = False):
    number_basis_functions = int(np.sqrt(len(w))-1);
    w_dot_p = np.zeros(w.shape);
    w_dot_n = np.zeros(w.shape);

    #Advection
    if usefft:
        flow = Reconstruct_Velocity(w, None, res=np.array([number_basis_functions+1, number_basis_functions+1]), usefft = True, res_modifier = 4);
        vortictiy = Reconstruct_Vorticity(w, number_basis_functions+1, number_basis_functions+1, res_modifier = 4);
        n_b_squared = len(w)
        helper = np.zeros([2, n_b_squared, n_b_squared]);
        for i1 in range(1, number_basis_functions+1):
            for i2 in range(1, number_basis_functions+1):
                helper[0,i1*(number_basis_functions+1)+i2,i1*(number_basis_functions+1)+i2] = -float(i2)**2/(float(i1)**2 + float(i2)**2)
                helper[1,i1*(number_basis_functions+1)+i2,i1*(number_basis_functions+1)+i2] = float(i1)**2/(float(i1)**2 + float(i2)**2);

        advection_x = -flow[1,:,:]*vortictiy;
        advection_y = flow[0,:,:]*vortictiy;

        projection_x, projection_y = Project_Velocity(advection_x, advection_y, number_basis_functions);

        advx = np.dot(helper[0], projection_x.reshape([n_b_squared]))
        advy = np.dot(helper[1], projection_y.reshape([n_b_squared]))

        #deriviation
        w_dotp_fft = -advx+advy;
        # w_dotp_fft[0,:] = 0;
        # w_dotp_fft[:,0] = 0;

        w_dot_p = w_dotp_fft.reshape(n_b_squared);
    else:
        w_dot_p = C.Apply(w);
    #Viscosity

    for i in range(len(w_dot_n)):
        w_dot_n[i] += viscosity*lambda_k(i/(number_basis_functions+1), i%(number_basis_functions+1))*w[i];

    #External forces

    if f != None:
        w_dot_n += f;

    return w_dot_p, w_dot_n;

def GetBasisfunctions(basis_count , x1 , x2 ):
    tmp = np.zeros([basis_count+1, basis_count+1, 2, x1.shape[0], x2.shape[1]]);
    for i in range(1, basis_count+1):
        for j in range(1, basis_count+1):
            tmp[i, j] = Basisfunction_Velocity(i, j, x1, x2);
    basisfunctions = tmp.reshape([(basis_count+1)**2, 2, x1.shape[0], x1.shape[1]]);
    return basisfunctions

def GetBasis_vortisity_functions(basis_count , x1 , x2 ):
    tmp = np.zeros([basis_count+1, basis_count+1, x1.shape[0], x2.shape[1]]);
    for i in range(1, basis_count+1):
        for j in range(1, basis_count+1):
            tmp[i,j] = Basisfunction_Vorticity(i, j, x1, x2);
    basisfunctions = tmp.reshape([(basis_count+1)**2, x1.shape[0], x2.shape[1]]);
    return basisfunctions;
