# global imports
import numpy as np
import sys

# local imports
sys.path.append('../') # add root code folder to search path for include of ../advection

# TODO: only load needed libraries
from advection.global_functions import *
from advection.integration import *
from advection.Log import *
import advection.parameter as pm

class data_generator:

    def __init__(self, steps, fps, M, N, dirPath, epsilon = 0.001, viscosity = 0, creational_steps=0, cache_size=1, val=False, train_steps=0):
        self.steps = steps
        self.fps = fps
        self.M = M
        self.N = N
        self.e = epsilon
        self.v = viscosity
        self.T = creational_steps
        self.train_steps = train_steps # in doubt use train_steps = T
        self.cache_size = cache_size
        print('generator started')
        self.val = val
        self.dirPath = dirPath

    def __call__(self):
        while True:
            # Take random slice from [0,T]
            if self.val:            
                random_flow_slice = np.random.randint(0, self.val_samples)
            else:
                random_flow_slice = np.random.randint(0, self.cache_size)
            T_at = np.random.randint(0, self.train_steps-self.steps-1)

            # Exctract window around random selected slice
            if self.val:
                steps_N = np.load(self.dirPath+"/val_{}.npy".format(random_flow_slice))[:,T_at:T_at+self.steps]
                step_M = np.load(self.dirPath+"/val_stepWise_M_{}.npy".format(random_flow_slice))[:,T_at+self.steps-1]
            else:
                steps_N = np.load(self.dirPath+"/train_{}.npy".format(random_flow_slice))[:,T_at:T_at+self.steps]
                step_M = np.load(self.dirPath+"/train_stepWise_M_{}.npy".format(random_flow_slice))[:,T_at+self.steps-1]

            # take M_basis from last step as target Y
            Y = np.copy(np.squeeze(steps_N[:,-1]))
            Y = np.reshape(Y, (self.N+1, self.N+1))
            Y = Y[1:self.M+1, 1:self.M+1]

            # create last integration step using only M_basis
            M_temp = np.reshape(step_M, (self.N+1, self.N+1))

            # take M_basis from steps_N
            X = np.copy(steps_N) 
            X = np.reshape(np.transpose(X), (self.steps, self.N+1, self.N+1))
            X = X[:, :self.M+1, :self.M+1]
            X[-1,:,:] = np.copy(M_temp[:self.M+1, :self.M+1])
            X = np.copy(X[:,1:,1:])
            
            yield  X, Y