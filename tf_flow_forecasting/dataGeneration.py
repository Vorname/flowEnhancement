#%%
# this file is solely for generation of training and validation data
# created data is stored in ./flowData
# ./flowData contains the generated flows as well as the parameter.py files used for creation

# parameters are set in the ../advection/parameter.py
# prerequisits: initial flow file via ../advection/compareBaseFlow.py 

import numpy as np
import sys
import time
import datetime
import multiprocessing as mp
from pathlib import Path



sys.path.append('../') # make advection available

from advection.integration import *
import advection.parameter as pm


#%%

# compute flow in N^2 for training and validation data:
def initializeTrainData(mat_list, a, b, numK, M, N, visc, T, fps, everyKnews, pid, cacheSize, dirPath):
    print("process {} to {} startet".format(a,b))
    for k in range(a, b):
        baseFlow_tmp = np.zeros(((N+1)**2, T))
        IntegrateSimple(mat_list[k,:], visc, T, fps, None, baseFlow_tmp)
        # write data to file
        if k >= cacheSize:
            np.save(dirPath+"/val_{}.npy".format(k-cacheSize), baseFlow_tmp)  
        else:
            np.save(dirPath+"/train_{}.npy".format(k), baseFlow_tmp)
        # compute stepWise flow M to speed up training and validation
        stepWise_M_flow = np.zeros(((N+1)**2, T))
        for t in range(1, T-1):
            stepWise_flow_M = baseFlow_tmp[:,t-1]
            stepWise_flow_M = np.reshape(np.reshape(stepWise_flow_M,(N+1, N+1))[:M+1,:M+1],((M+1)**2))
            tmp_M = np.zeros(((M+1)**2,2))
            IntegrateSimple(stepWise_flow_M, visc, 2, fps, None, tmp_M)
            # Insert into N**2 basis for compatibility
            flow_N = np.zeros((N+1, N+1))
            flow_N[:M+1, :M+1] = np.copy(np.reshape(tmp_M[:,-1],(M+1, M+1)))
            flow_N = np.reshape(flow_N,(N+1)**2)
            stepWise_M_flow[:,t] = np.copy(flow_N)
        if k >= cacheSize:
            np.save(dirPath+"/val_stepWise_M_{}.npy".format(k-cacheSize), stepWise_M_flow)  
        else:
            np.save(dirPath+"/train_stepWise_M_{}.npy".format(k), stepWise_M_flow)
        if np.mod(k,everyKnews)==0:
            print('pid:{}, k:{}/{} done'.format(pid,k,b))
    print("process {} to {} finished".format(a,b))


if __name__ == '__main__':
    timer_start = time.time()
    num_flows = pm.cache_size+pm.val_samples

    # create dir to store data in:
    dirPath = "./flowData/M{}_N{}_C{}_S{}_V{}_E{}".format(pm.basis_M, pm.basis_N, pm.creational_steps, pm.cache_size, pm.val_samples, pm.eps)
    Path(dirPath).mkdir(parents=True, exist_ok=True)

    # load inital flow (base flow)
    initial_flow = np.squeeze( np.reshape(np.load(pm.initialFlowFile),((pm.basis_N+1)**2)))
    # create a list to store perptated flows
    mat_list = np.zeros((num_flows,(pm.basis_N+1)**2))
    # perpuate flows and store in list
    for k in range(num_flows):
        # initialize random epsilon flow to offset inital flow with
        random_e_flow = np.zeros([pm.basis_N+1, pm.basis_N+1])
        random_e_flow[1:pm.basis_M+1, 1:pm.basis_M+1] = (np.random.rand(pm.basis_M, pm.basis_M)-0.5)*2 # goes from -1 to 1
        random_e_flow = random_e_flow.reshape((pm.basis_N+1)*(pm.basis_N+1))
        # Offeset every base flow expect the first 3 one
        if k == 0 or k == pm.cache_size:
            # base flow unperpuated MUST be included in train and val data for convience
            flow_N = initial_flow
        else:
            flow_N = initial_flow + pm.eps * random_e_flow
        mat_list[k,:] = flow_N

    # compute the N basis integration
    print('epsilon initialized')
    processes = mp.cpu_count()
    print('starting parallel data computation using {} cpu cores'.format(processes))
    
    count = min(processes, num_flows)
    p = [mp.Process(target=initializeTrainData, args = (mat_list, int(i*(num_flows/count)), int((i+1)*(num_flows/count)), num_flows, pm.basis_M, pm.basis_N, pm.visc, pm.creational_steps, pm.fps, num_flows/100, i, pm.cache_size, dirPath)) for i in range(count)]
    for pp in p:
        pp.start()
    for pp in p:
        pp.join()

    timer_end = time.time()
    print("Creation of N^2 data took {}sec\n".format(timer_end-timer_start))


    print('computing stepWise and iterative M flows:')

    timer_end = time.time()
    print("Creation data took {}sec\n".format(timer_end-timer_start))