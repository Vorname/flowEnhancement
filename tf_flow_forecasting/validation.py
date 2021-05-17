# this file is solely for validation of a given network stored in ./trainingSets
# results are stored in the input directory ./trainingSets/* to ease work flow
# ./trainingSets/* gets validationData used for plotting

# parameters are set from the specified trainingSet, the model and inputData must be specified there

# prerequisits: complete set in ./trainingSet with available data
import tensorflow as tf
import numpy as np
import sys
import multiprocessing as mp
import time
from pathlib import Path

# local imports
sys.path.append('../') # add root code folder to search path for includes

# Import parameters
import advection.parameter as pm
#from makeCustomModel import *
from advection.integration import *

def computeIterativeValDataGT(a, b, cacheSize, valSamples, M, N, T, S, iterOffset, visc, fps, dirPath, modelPath):
    print("process {} to {} startet".format(a,b))
    # k goes from 0:valSamples
    for k in range(a, b):
        valSample = np.load(dirPath+"/val_{}.npy".format(k))
        # compute iterative flow unsing only M^2 basis for total integration time T
        tmp_M_flow_iterative = np.zeros(((M+1)**2, T))
        iterative_M_flow_zeroAppended = np.zeros((T, N+1, N+1))
        iterative_flow_M = np.zeros((M+1, M+1))
        #TODO: S+iterOffset, why S? Should start at zero?
        iterative_flow_M[:M+1, :M+1] = np.reshape(valSample[:, S+iterOffset-1], (N+1,N+1))[:M+1,:M+1]
        iterative_flow_M = np.reshape(iterative_flow_M,((M+1)**2))
        IntegrateSimple(iterative_flow_M, visc, T-iterOffset, fps, None, tmp_M_flow_iterative[:,iterOffset:])
        iterative_M_flow_zeroAppended[:,:M+1,:M+1] = np.reshape(np.swapaxes(tmp_M_flow_iterative,0,1), (T, M+1, M+1))
        iterative_M_flow_zeroAppended = np.swapaxes(np.reshape(iterative_M_flow_zeroAppended, (T,(N+1)**2)),0,1)
        iterative_M_flow_zeroAppended[:,:iterOffset] = valSample[:,:iterOffset]
        np.save(modelPath+"/iterativeGT{}/val_iterative_M_{}.npy".format(iterOffset,k), iterative_M_flow_zeroAppended)
        print("k {} done".format(k))

# Input requirement:
# val data 
# Gives:
# iterative_C_fake
def computeIterativeCorrectedFlowFake(valSamples, M, N, T, S, SOffset, modelPath, pathToTrainData, visc, fps):
    # Set CPU as available physical device
    #tf.config.set_visible_devices([], 'GPU')
    model = tf.keras.models.load_model(modelPath+'/trainedModel')
    # simply load the validation data computed on N basis and overwrite it (eases k=0 creation of input data)
    for k in range(valSamples):
        val_iterative_C = np.load(pathToTrainData+"/val_{}.npy".format(k))
        val_stepWise_M = np.load(pathToTrainData+"/val_stepWise_M_{}.npy".format(k))
        # reverence flow precomputed by dataGeneration
        for t in range(S+SOffset-1, T):
            nextInput = np.empty((1, S, M, M))
            nextInput[0,:,:,:] = np.copy(np.reshape(np.transpose(val_iterative_C[:,t-S+1:t+1]),(S,N+1,N+1))[:,1:M+1,1:M+1])
            M_tmp = val_stepWise_M[:,t]
            if not np.all(np.logical_not(np.isnan(M_tmp))):
                print(flow_M)
                print(M_tmp)
                print('at i:{} prediction contains nan'.format(i))
                break
            nextInput[0,-1,:,:] = np.copy(np.reshape(M_tmp,(N+1,N+1))[1:M+1,1:M+1])
            nextInput = tf.data.Dataset.from_tensor_slices((nextInput, None))
            nextInput = nextInput.batch(1)

            # Correct previous integration step using lstm
            tmp = np.zeros((N+1,N+1))
            #print('prediction!')
            tmp[1:M+1,1:M+1] = np.squeeze(model.predict(nextInput))
            currentN = np.reshape(val_iterative_C[:,t],(N+1,N+1))[:M+1,:M+1]
            #print("Energy                                       : {}".format(np.sum((np.reshape(val_iterative_C[:,t],(N+1,N+1))[:M+1,:M+1])**2) ))
            #print("Distance gt to integration(C)                : {}".format(np.sum((currentN - np.reshape(M_tmp,(N+1,N+1))[:M+1,:M+1])**2))) # should be small and is in the same time frame if everything is correct
            #print("Distance gt to enhancement(integration(C))   : {}".format(np.sum((currentN - np.reshape(tmp,(N+1,N+1))[:M+1,:M+1])**2))) # distance should shrink
            #print("Distance gt to ideal input                   : {}".format(np.sum((currentN - np.reshape(val_stepWise_M[:,t],(N+1,N+1))[:M+1,:M+1])**2))) # distance gt M to N
            #print("Distance itegration(C) to input              : {}".format(np.sum((np.reshape(M_tmp,(N+1,N+1))[:M+1,:M+1] - np.reshape(val_stepWise_M[:,t],(N+1,N+1))[:M+1,:M+1])**2))) # distance gt M to N
            val_iterative_C[:, t] = np.copy(tmp.flatten())
            # remove the breaks (only debug)
            #if t == S+SOffset-1 + 80:
            #    print("debug exit!")
            #    exit()
        np.save(modelPath+"/iterativeFake{}/val_iterative_C_{}.npy".format(SOffset,k), val_iterative_C)
        print('iterative forecasting for k {} done'.format(k))

# Input requirement:
# val data 
# Gives:
# iterative_C_true
def computeIterativeCorrectedFlowTrue(engConv, valSamples, M, N, T, S, SOffset, modelPath, pathToTrainData, visc, fps):
    # Set CPU as available physical device
    #tf.config.set_visible_devices([], 'GPU')
    model = tf.keras.models.load_model(modelPath+'/trainedModel')
    # simply load the validation data computed on N basis and overwrite it (eases k=0 creation of input data)
    for k in range(valSamples):
        val_iterative_C = np.load(pathToTrainData+"/val_{}.npy".format(k))
        val_stepWise_M = np.load(pathToTrainData+"/val_stepWise_M_{}.npy".format(k))
        val_N = np.load(pathToTrainData+"/val_{}.npy".format(k))
        # reverence flow precomputed by dataGeneration
        E_out = 0
        E_in  = 0
        tmp = np.zeros((N+1,N+1))
        for t in range(S+SOffset-1, T):
            nextInput = np.empty((1, S, M, M))
            nextInput[0,:,:,:] = np.copy(np.reshape(np.transpose(val_iterative_C[:,t-S+1:t+1]),(S,N+1,N+1))[:,1:M+1,1:M+1])
            # M_temp needs to be integrated using M^2 basis and tmp here! 
            if t == S+SOffset-1:
                M_onC = np.copy(np.reshape(val_stepWise_M[:,t],(N+1,N+1))[:M+1,:M+1])
            else:
                M_onC = np.zeros(((M+1)**2,2))

                # CAVEAT! Here happens important stuff!
                #integrationInput = np.reshape(np.reshape(val_N[:,t-1],(N+1, N+1))[:M+1,:M+1],((M+1)**2))
                integrationInput = np.copy((tmp[:M+1,:M+1]).flatten())
                if engConv == 1:
                    E_in = np.sum(np.reshape(np.reshape(val_N[:,t-1],(N+1, N+1))[:M+1,:M+1],((M+1)**2))**2)
                if engConv == 2:
                    E_in = np.sum(integrationInput**2)

                IntegrateSimple(integrationInput.flatten() , visc, 2, fps, None, M_onC)
                M_onC = np.copy(M_onC[:,-1])
                E_out =  np.sum(M_onC**2)# this uses the energy of the true case...
                M_onC = np.reshape(M_onC,(M+1,M+1))
                if engConv > 0:
                    M_onC = M_onC*np.sqrt(E_in/E_out) # introduces havoc! No the opposite!

                #M_onC = integrationInput # brute force the fake integratoin behaviour for deubg
            if not np.all(np.logical_not(np.isnan(M_onC))):
                print('at i:{} prediction contains nan'.format(t))
                val_iterative_C[:,t:] = np.nan
                break
            #    exit()
        
            nextInput[0,-1,:,:] = np.copy(M_onC[1:,1:])
            nextInput = tf.data.Dataset.from_tensor_slices((nextInput, None))
            nextInput = nextInput.batch(1)
            tmp[1:M+1,1:M+1] = np.squeeze(model.predict(nextInput))
            # might need stop condition to avoid explosion of energy, distance or complete nonesense in integration

            # debug
            currentN = np.reshape(val_iterative_C[:,t],(N+1,N+1))[:M+1,:M+1]
            #print("Energy                                       : {}".format(np.sum((tmp[:M+1,:M+1])**2)))
            #print("Distance gt to integration(C)                : {}".format(np.sum((currentN - M_onC)**2))) # should be small and is in the same time frame if everything is correct
            #print("Distance gt to enhancement(integration(C))   : {}".format(np.sum((currentN - np.reshape(tmp,(N+1,N+1))[:M+1,:M+1])**2))) # distance should shrink
            #print("Distance gt to ideal input                   : {}".format(np.sum((currentN - np.reshape(val_stepWise_M[:,t],(N+1,N+1))[:M+1,:M+1])**2))) # should be independet regardless of other computations
            #print("Distance itegration(C) to input              : {}".format(np.sum((M_onC - np.reshape(val_stepWise_M[:,t],(N+1,N+1))[:M+1,:M+1])**2))) # distance gt M to N
            
            val_iterative_C[:, t] = np.copy(tmp.flatten())

            # remove the breaks (only debug)
            #if t == S+SOffset-1 + 80:
            #    print("debug exit!")
            #    exit()
        if engConv == 0:
            np.save(modelPath+"/iterativeTrue{}/val_iterative_C_NoConv_{}.npy".format(SOffset,k), val_iterative_C)
        if engConv == 1:
            np.save(modelPath+"/iterativeTrue{}/val_iterative_C_FakeConv_{}.npy".format(SOffset,k), val_iterative_C)
        if engConv == 2:
            np.save(modelPath+"/iterativeTrue{}/val_iterative_C_TrueConv_{}.npy".format(SOffset,k), val_iterative_C)
        print('iterative forecasting for k {} done'.format(k))

# Input requirement:
# val
# stepWise M 
# Output:
# stepWise_C
def computeStepWiseValFlow(valSamples, basisM, basisN, T, inSteps, modelPath, pathToData, visc, fps):
    #tf.config.set_visible_devices([], 'GPU')
    model = tf.keras.models.load_model(modelPath+'/trainedModel')

    for k in range(valSamples):
        # load val data
        valFlow_N = np.load(pathToData+"/val_{}.npy".format(k))
        valFlow_M = np.load(pathToData+"/val_stepWise_M_{}.npy".format(k)) 
        valFlow_C_stepwise = np.zeros(((basisN+1)**2, T))
        valFlow_Dist_C_stepwise_Integrated_onC = np.zeros((pm.creational_steps))
        valFlow_Dist_C_stepwise_Integrated_onM = np.zeros((pm.creational_steps))
        # prediction needs pm.steps as input to work, fill past steps with true data
        valFlow_C_stepwise[:,:pm.steps] = valFlow_N[:,:pm.steps]
        for t in range(pm.steps, pm.creational_steps+1):
            # get last steps and create input

            nextInput = np.empty((1, pm.steps, pm.basis_M, pm.basis_M))
            for i in range(pm.steps-1):
                nextInput[0,i,:,:] = np.copy(np.reshape(valFlow_N[:,t-pm.steps+i], (pm.basis_N+1,pm.basis_N+1)))[1:pm.basis_M+1,1:pm.basis_M+1]
            nextInput[0,-1,:,:] = np.copy(np.reshape(valFlow_M[:,t-1], (pm.basis_N+1,pm.basis_N+1))[1:pm.basis_M+1,1:pm.basis_M+1])
            nextInput = tf.data.Dataset.from_tensor_slices((nextInput, None))
            nextInput = nextInput.batch(1)
            prediction = model.predict(nextInput)
            # induce into zero (N+1)^2
            flow_N = np.zeros((pm.basis_N+1, pm.basis_N+1))
            flow_N[1:pm.basis_M+1,1:pm.basis_M+1] = np.reshape(np.squeeze(prediction),(pm.basis_M, pm.basis_M))
            #valFlow_C_stepwise[:,t-1] = np.copy(np.reshape(flow_N,((pm.basis_N+1)**2)))

            # integrate prediction using M basis and compare distances to val
            M_AdvLstmM = np.zeros(((pm.basis_M+1)**2,2))
            integrationInput = np.copy(flow_N[:pm.basis_M+1,:pm.basis_M+1])
            IntegrateSimple(integrationInput.flatten() , visc, 2, fps, None, M_AdvLstmM)
            M_AdvLstmM = np.copy(M_AdvLstmM[:,-1])

            # integrate last input step using M basis for comparision
            M_AdvM = np.zeros(((pm.basis_M+1)**2,2))
            integrationInput = np.copy(np.reshape(valFlow_M[:,t-1], (pm.basis_N+1,pm.basis_N+1))[:pm.basis_M+1,:pm.basis_M+1])
            IntegrateSimple(integrationInput.flatten() , visc, 2, fps, None, M_AdvM)
            M_AdvM = np.copy(M_AdvM[:,-1])
            
            if t < pm.creational_steps:
                valFlow_Dist_C_stepwise_Integrated_onC[t] = np.sum((M_AdvLstmM - (np.reshape(valFlow_N[:,t], (pm.basis_N+1, pm.basis_N+1))[:pm.basis_M+1,:pm.basis_M+1]).flatten())**2)
                valFlow_Dist_C_stepwise_Integrated_onM[t] = np.sum((M_AdvM - (np.reshape(valFlow_N[:,t], (pm.basis_N+1, pm.basis_N+1))[:pm.basis_M+1,:pm.basis_M+1]).flatten())**2)
            #print("Distance  AdvLstmM to gt: {}".format(np.sum((M_AdvLstmM - (np.reshape(valFlow_N[:,t], (pm.basis_N+1, pm.basis_N+1))[:pm.basis_M+1,:pm.basis_M+1]).flatten())**2)))
            #print("Distance      AdvM to gt: {}".format(np.sum((M_AdvM - (np.reshape(valFlow_N[:,t], (pm.basis_N+1, pm.basis_N+1))[:pm.basis_M+1,:pm.basis_M+1]).flatten())**2)))

            # For TESTING: do nothing -> M and C step solution should be equal
            valFlow_C_stepwise[:,t-1] = np.copy(np.reshape(flow_N,((pm.basis_N+1)**2)))
        #print("MSE distance AdvLstm, AdvM to gt: {}".format(np.mean(valFlow_Dist_C_stepwise_Integrated_onC - valFlow_Dist_C_stepwise_Integrated_onM)))
        #print("Max distance AdvLstm, AdvM to gt: {}".format(np.max(valFlow_Dist_C_stepwise_Integrated_onC - valFlow_Dist_C_stepwise_Integrated_onM)))
        np.save(modelPath+"/stepWise/val_stepWise_C_{}.npy".format(k), valFlow_C_stepwise)
        np.save(modelPath+"/stepWise/val_stepWise_C_Integrated_on_C{}.npy".format(k), valFlow_Dist_C_stepwise_Integrated_onC)
        np.save(modelPath+"/stepWise/val_stepWise_C_Integrated_on_M{}.npy".format(k), valFlow_Dist_C_stepwise_Integrated_onM)
        print("step wise correction for k {} done".format(k))

#advLstmM = np.load("./trainingSets/run800debug/stepWise/val_stepWise_C_Integrated_on_C0.npy")

if __name__ == '__main__':
    timer_start = time.time()

    stepwise      = False #pm.computeStepwise
    iterativeGT   = True #
    iterativeFake = False #
    iterativeTrue = True #
    print('what in the fuck')

    print(pm.ModelPath)

    pathToTrainData = "./flowData/M{}_N{}_C{}_S{}_V{}_E{}".format(pm.basis_M, pm.basis_N, pm.creational_steps, pm.cache_size, pm.val_samples, pm.eps)
    print("loading data from: {}".format(pathToTrainData))

    Path(pm.ModelPath+"/stepWise").mkdir(parents=True, exist_ok=True)

        
    if stepwise:
        print('computing valFlow_C_stepwise')
        computeStepWiseValFlow(pm.val_samples, pm.basis_M, pm.basis_N, pm.creational_steps, pm.steps, pm.ModelPath, pathToTrainData, pm.visc, pm.fps)
    
    timer_between = time.time()
    print("Validation StepWise took {}sec\n".format(timer_between-timer_start))

    for Soff in range(pm.numOffsets):
        Path(pm.ModelPath+"/iterativeFake{}".format(Soff*pm.fps)).mkdir(parents=True, exist_ok=True)
        Path(pm.ModelPath+"/iterativeTrue{}".format(Soff*pm.fps)).mkdir(parents=True, exist_ok=True)
        Path(pm.ModelPath+"/iterativeGT{}".format(Soff*pm.fps)).mkdir(parents=True, exist_ok=True)

        if iterativeGT:
            print('starting iterative gt creation in paralell')
            count = min(mp.cpu_count(), pm.val_samples)
            p = [mp.Process(target=computeIterativeValDataGT, args = (int(i*(pm.val_samples/count)), int((i+1)*(pm.val_samples/count)), pm.cache_size, pm.val_samples, pm.basis_M, pm.basis_N, pm.creational_steps, pm.steps, Soff*pm.fps, pm.visc, pm.fps, pathToTrainData, pm.ModelPath)) for i in range(count)]
            for pp in p:
                pp.start()

            for pp in p:
                pp.join()

        if iterativeFake:
            print('starting  iterative fake correction')
            computeIterativeCorrectedFlowFake(pm.val_samples, pm.basis_M, pm.basis_N, pm.creational_steps, pm.steps, Soff*pm.fps, pm.ModelPath, pathToTrainData, pm.visc, pm.fps)

        # currently not really working
        if iterativeTrue:
            print('starting  iterative true correction')
            # no energy conservation
            computeIterativeCorrectedFlowTrue(0, pm.val_samples, pm.basis_M, pm.basis_N, pm.creational_steps, pm.steps, Soff*pm.fps, pm.ModelPath, pathToTrainData, pm.visc, pm.fps)
            # fake energy conservation
            computeIterativeCorrectedFlowTrue(1, pm.val_samples, pm.basis_M, pm.basis_N, pm.creational_steps, pm.steps, Soff*pm.fps, pm.ModelPath, pathToTrainData, pm.visc, pm.fps)
            # true energy conservation
            computeIterativeCorrectedFlowTrue(2, pm.val_samples, pm.basis_M, pm.basis_N, pm.creational_steps, pm.steps, Soff*pm.fps, pm.ModelPath, pathToTrainData, pm.visc, pm.fps)

    
    timer_end = time.time()
    print("Validation Iterative took {}sec\n".format(timer_end-timer_between))
    print("Total Validation time: {}sec".format(timer_end-timer_start))