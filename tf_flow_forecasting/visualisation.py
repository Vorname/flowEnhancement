#%%
# this file is solely to visualize various things (trainingData, validationData, validationResults, model architecture)
# visualizations can be stored together with trained model in ./trainingSets/*
# adds plots to ./trainingSets/* or plots them directly to browser

# parameters are taken from specified ./trainingSets/*
# prerequisits: depends on situation, mostly used interactive after successfull data generation or model validation

import numpy as np
import sys
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

sys.path.append('../') #

import advection.parameter as pm
from plottingFun import *

def extractM(flow, K, T, M, N):
    ret = np.copy(np.reshape(np.swapaxes(flow,1,2), (K, T, N+1, N+1)))
    ret[:,:,M+1:,:] = 0
    ret[:,:,:,M+1:] = 0
    ret = np.copy(np.swapaxes(np.reshape(ret, (K, T, (N+1)**2)),1,2))
    return ret

if __name__ == "__main__":
    num_val_steps = pm.creational_steps # TODO: impelement to make scores based on train data length and more (differenciate)

    # set plotting behaviour
    plotFlowData = True
    fullBandwidth = False # plot up to M or up to N basis
    colorArray2 = ["black", "#ED6190"] #["black", "#5DC9F0", "#658FF7", "#7B67E0", "#C265F7", "#ED61D6", "#ED6190"] # made with adobe oclorpicker
    colorArray3 = ["black", "black", "#ED6190"] #["black", "#5DC9F0", "#658FF7", "#7B67E0", "#C265F7", "#ED61D6", "#ED6190"] # made with adobe oclorpicker
    colorArray4 = ["#5DC9F0", "black", "#ED6190"] #["black", "#5DC9F0", "#658FF7", "#7B67E0", "#C265F7", "#ED61D6", "#ED6190"] # made with adobe oclorpicker
    numLines = 10 # number of individually plottet flows
    normalize = True # normalize energy bands using (2*l-1) as factor
    axisEqual = False # force the same axis scaling for all energyband plots
    stopAt = 150

    scoringOnly = False
    plotScores  = True
    plotEnery   = True
    plotCoeffs  = False
    writeFigures = False

    # load the data
    # Aims to have all the data in format: [numFlows, (N+1)^2, T] (fft zeros embeded)
    # For M flows this results in matrices containing zeros at [:,>M,:]
    valData          = np.zeros((pm.val_samples, (pm.basis_N+1)**2, pm.creational_steps))
    stepWise_M       = np.zeros((pm.val_samples, (pm.basis_N+1)**2, pm.creational_steps))
    iterative_M      = np.zeros((pm.val_samples, (pm.basis_N+1)**2, pm.creational_steps))
    stepWise_C       = np.zeros((pm.val_samples, (pm.basis_N+1)**2, pm.creational_steps))
    stepWise_C_ItrC  = np.zeros((pm.val_samples, pm.creational_steps))
    stepWise_C_ItrM  = np.zeros((pm.val_samples, pm.creational_steps))
    iterative_C_True_NoConv = np.zeros((pm.val_samples, (pm.basis_N+1)**2, pm.creational_steps))


    trainDataPath = "./flowData/M{}_N{}_C{}_S{}_V{}_E{}".format(pm.basis_M, pm.basis_N, pm.creational_steps, pm.cache_size, pm.val_samples, pm.eps)

    print("loading data from: {}".format(trainDataPath))

    #%%
    # iteratively:
    for k in range(pm.val_samples):
        valData[k,:,:]                   = np.load(trainDataPath+"/val_{}.npy".format(k)) #  from data
        stepWise_M[k,:,:]                = np.load(trainDataPath+"/val_stepWise_M_{}.npy".format(k)) #  from data
        stepWise_C[k,:,:]                = np.load(pm.ModelPath+"/stepWise/val_stepWise_C_{}.npy".format(k)) #  from trainSet
        stepWise_C_ItrC[k,:]           = np.load(pm.ModelPath+"/stepWise/val_stepWise_C_Integrated_on_C{}.npy".format(k)) # Adv(Lstm(w))
        stepWise_C_ItrM[k,:]           = np.load(pm.ModelPath+"/stepWise/val_stepWise_C_Integrated_on_M{}.npy".format(k)) # Two M-step integrations Adv(Adv(w))
        iterative_M[k,:,:]               = np.load(pm.ModelPath+"/iterativeGT{}/val_iterative_M_{}.npy".format(pm.valStepOffset,k)) #  from trainSet
        iterative_C_True_NoConv[k,:,:]   = np.load(pm.ModelPath+"/iterativeTrue{}/val_iterative_C_NoConv_{}.npy".format(pm.valStepOffset, k)) 
    print("loading done")

    valData          = extractM(valData, pm.val_samples, pm.creational_steps, pm.basis_M, pm.basis_N)
    stepWise_M       = extractM(stepWise_M, pm.val_samples, pm.creational_steps, pm.basis_M, pm.basis_N)
    iterative_M      = extractM(iterative_M, pm.val_samples, pm.creational_steps, pm.basis_M, pm.basis_N)
    stepWise_C       = extractM(stepWise_C, pm.val_samples, pm.creational_steps, pm.basis_M, pm.basis_N)
    iterative_C_True_NoConv   = extractM(iterative_C_True_NoConv, pm.val_samples, pm.creational_steps, pm.basis_M, pm.basis_N)

    # dirty fix the last step(s)
    valData[:,:,-1]                 = valData[:,:,-2]
    stepWise_M[:,:,-1]              = stepWise_M[:,:,-2]
    iterative_M[:,:,-1]             = iterative_M[:,:,-2]
    stepWise_C[:,:,-1]              = stepWise_C[:,:,-2]
    iterative_C_True_NoConv[:,:,-1] = iterative_C_True_NoConv[:,:,-2]
    # and the first one as well...
    valData[:,:,0]                 = valData[:,:,1]
    stepWise_M[:,:,0]              = stepWise_M[:,:,1]
    iterative_M[:,:,0]             = iterative_M[:,:,1]
    stepWise_C[:,:,0]              = stepWise_C[:,:,1]
    iterative_C_True_NoConv[:,:,0] = iterative_C_True_NoConv[:,:,1]

    print("reshape done")

    dist_stepWise_M_N                = np.zeros((pm.val_samples,pm.creational_steps))
    dist_stepWise_C_N                = np.zeros((pm.val_samples,pm.creational_steps))
    dist_stepWise_ItrC                = np.zeros((pm.val_samples,pm.creational_steps))
    dist_stepWise_ItrM                = np.zeros((pm.val_samples,pm.creational_steps))
    dist_iterative_M_N               = np.zeros((pm.val_samples,pm.creational_steps))
    dist_iterative_C_N_True_NoConv_MN   = np.zeros((pm.val_samples,pm.creational_steps))

    for k in range(pm.val_samples):
        for i in range(pm.basis_M+1):
            for j in range(pm.basis_M+1):
                # CAVEAT: introducing root to get distance from square distance, was not before
                dist_iterative_M_N[k,:]               = dist_iterative_M_N[k,:] + (valData[k,i*(pm.basis_N+1)+j,:]-iterative_M[k,i*(pm.basis_N+1)+j,:])**2
                #dist_iterative_C_N_Fake[k,:]          = dist_iterative_C_N_Fake[k,:] + (valData[k,i*(pm.basis_N+1)+j,:]-iterative_C_Fake[k,i*(pm.basis_N+1)+j,:])**2
                dist_iterative_C_N_True_NoConv_MN[k,:]   = dist_iterative_C_N_True_NoConv_MN[k,:] + (valData[k,i*(pm.basis_N+1)+j,:]-iterative_C_True_NoConv[k,i*(pm.basis_N+1)+j,:])**2
                dist_stepWise_M_N[k,:]                = dist_stepWise_M_N[k,:]  + (valData[k,i*(pm.basis_N+1)+j,:]-stepWise_M[k,i*(pm.basis_N+1)+j,:])**2
                dist_stepWise_C_N[k,:]                = dist_stepWise_C_N[k,:]  + (valData[k,i*(pm.basis_N+1)+j,:]-stepWise_C[k,i*(pm.basis_N+1)+j,:])**2
        dist_iterative_M_N[k,:]               = np.sqrt(dist_iterative_M_N[k,:])
        dist_iterative_C_N_True_NoConv_MN[k,:]   = np.sqrt(dist_iterative_C_N_True_NoConv_MN[k,:]) 
        dist_stepWise_M_N[k,:]                = np.sqrt(dist_stepWise_M_N[k,:])
        dist_stepWise_C_N[k,:]                = np.sqrt(dist_stepWise_C_N[k,:])
        dist_stepWise_ItrC[k,:]                = np.sqrt(stepWise_C_ItrC[k,:]) # already in square distance loaded
        dist_stepWise_ItrM[k,:]                = np.sqrt(stepWise_C_ItrM[k,:]) # already in square distance loaded

    print("distances computed")
    score_stepWise_CN = np.mean(dist_stepWise_C_N) # weak, does not take input flow into account! Use dotproduct or similar, dotproduct can be zero if C_N an M_N are paralell...
    score_stepWise_MN = np.mean(dist_stepWise_M_N) # weak, does not take input flow into account! Use dotproduct or similar, dotproduct can be zero if C_N an M_N are paralell...
    score_stepWise_ItrC = np.mean(dist_stepWise_ItrC) # weak, does not take input flow into account! Use dotproduct or similar, dotproduct can be zero if C_N an M_N are paralell...
    score_stepWise_IterM = np.mean(dist_stepWise_ItrM) # weak, does not take input flow into account! Use dotproduct or similar, dotproduct can be zero if C_N an M_N are paralell...
    score_iterative_True_NoConv_CN = np.mean(dist_iterative_C_N_True_NoConv_MN)
    score_iterative_True_NoConv_MN = np.mean(dist_iterative_M_N)

    print("scores computed:\n")

    f = open("./FinalScores07.04.21.csv", "a+")
    f.write("\n{}&{}&{}&{}&{}&{}&{}".format(pm.run, score_stepWise_MN, score_stepWise_CN, score_iterative_True_NoConv_MN, score_iterative_True_NoConv_CN, score_stepWise_IterM, score_stepWise_ItrC)) # Scores are stepWise [N-M,N-C] and iterativeNoConv [N-M,N-C] and 2step Iter on [M,C]
    f.close()
    if scoringOnly:
        print('early exit due to scoringOnly')
        exit()

    if plotScores:

        fig_stepWise_comparison = plotDistance([dist_stepWise_M_N, dist_stepWise_C_N], colorArray2,title="Step wise flow M to N and C to N distances ({}) ".format(pm.ModelPath))
        fig_iterative_comparison_True_NoConv = plotDistance([dist_iterative_M_N, dist_iterative_C_N_True_NoConv_MN], colorArray2,title="Distance from interative true NO M to N and C to N flow ({}) ".format(pm.ModelPath))
        fig_IterStp = plotDistance([dist_stepWise_ItrM, dist_stepWise_ItrC], colorArray2, title="Distance from Adv(Adv(w)) and Adv(N(w)) to N ({}) ".format(pm.ModelPath))
        fig_iterative_comparison_True_NoConv_Short = plotDistance([dist_iterative_M_N, dist_iterative_C_N_True_NoConv_MN], colorArray2, stopAt=stopAt, title="Distance from interative true NO M to N and C to N flow ({}) ".format(pm.ModelPath))

        fig_stepWise_comparison.show()
        fig_iterative_comparison_True_NoConv.show()
        fig_IterStp.show()
        fig_iterative_comparison_True_NoConv_Short.show()
        if writeFigures:
            fig_stepWise_comparison.write_image("../../writing/thesis/diagrams/VAL_stp_cmp.svg")
            fig_iterative_comparison_True_NoConv.write_image("../../writing/thesis/diagrams/VAL_itr_true_no_cmp.svg")
            fig_iterative_comparison_True_NoConv_Short.write_image("../../writing/thesis/diagrams/VAL_itr_true_no_cmp_SHORT.svg")
            fig_IterStp.write_image("../../writing/thesis/diagrams/VAL_ItrStp.svg")
        
        # plot wave number based stuff
    if plotEnery:
        # depends on M
        # Energy will be higher in 
        #testData   = np.zeros((pm.val_samples, pm.basis_N+1, pm.basis_N+1, pm.creational_steps))
        #testData[:,1:,1:,:] = 1
        #testData = np.reshape(testData,(pm.val_samples, (pm.basis_N+1)**2, pm.creational_steps))
        print("starting energy computation")
        fig_energy_cmp  = plotEnergy([valData, stepWise_M, stepWise_C], colorArray4, pm.val_samples, pm.basis_M, pm.basis_N, pm.creational_steps)
        fig_energy_cmp.show()
        exit()
        fig_energy_in_L = plotBandEnergyDistance([valData, stepWise_M, stepWise_C], colorArray3, pm.val_samples, pm.basis_M, pm.basis_N, pm.creational_steps, axisEqual=axisEqual, normalize=normalize, title='stepWise energy difference bands for N (target) to M and C flows ({})'.format(pm.ModelPath))
        fig_energy_in_L.show()
        if writeFigures:
            fig_energy_in_L.write_image("../../writing/thesis/diagrams/VAL_Energy_StpWise.svg")
        fig_energy_in_L = plotBandEnergyDistance([valData, iterative_M, iterative_C_True_NoConv], colorArray3, pm.val_samples, pm.basis_M, pm.basis_N, pm.creational_steps, axisEqual=False, normalize=normalize, stopAt=stopAt, title='iterative energy difference bands for N (target) to M and C flows ({})'.format(pm.ModelPath))
        fig_energy_in_L.show()
        if writeFigures:
            fig_energy_in_L.write_image("../../writing/thesis/diagrams/VAL_Energy_Iterative_True_NoConv.svg")

    #%%
    if plotCoeffs:
        # plot coeff wise. Not working fo M=20..
        # make this work only on M^2 to compute distance
        flowDiff_M_N = valData-stepWise_M # zero best
        #print(np.reshape(flowDiff_M_N[0,:,:],(pm.basis_N+1,pm.basis_N+1,pm.creational_steps))[1:pm.basis_M+1,1:pm.basis_M+1,:])
        flowDiff_C_N = valData-stepWise_C # zero best
        print('start plotting')
        fig_stepWise = plotCoeffs([valData, stepWise_M, stepWise_C], pm.basis_M, pm.basis_N, ['blue','black','red'], numLines=10,title="Step wise w in N bases, M integration, and corrected M integration ({}) ".format(pm.ModelPath))
        print('step wise done')
        fig_stepWiseDist_M_N = plotCoeffs([flowDiff_M_N], pm.basis_M, pm.basis_N, ['black'], numLines=10, plotMaxima=True, singleScale=False,title="Step wise distance from M steps to N ({}) ".format(pm.ModelPath))
        print('step wise MN done')
        fig_stepWiseDist_C_N = plotCoeffs([flowDiff_C_N], pm.basis_M, pm.basis_N, ['red'], numLines=10, plotMaxima=True, singleScale=False,title="Step wise distance from C steps to N ({}) ".format(pm.ModelPath))
        print('step wise CN done')
        fig_stepWiseDist_C_M = plotCoeffs([stepWise_C-stepWise_M], pm.basis_M, pm.basis_N, ['green'], numLines=10, plotMaxima=True, singleScale=False,title="Step wise distance from C steps to M ({}) ".format(pm.ModelPath))
        print('step wise CM done')
        fig_stepWiseDist_DistDiff = plotCoeffs([np.abs(flowDiff_C_N)-np.abs(flowDiff_M_N)], pm.basis_M, pm.basis_N, ['green'], numLines=10, plotMaxima=True, singleScale=False, ephasizeBaseFlow=False, highLightMean=True,title="Step wise dist diff (negative is better) ({}) ".format(pm.ModelPath))
        print('step dist diff done')

        fig_iterative = plotCoeffs([valData, iterative_C_Fake, iterative_M], pm.basis_M, pm.basis_N, ['blue','red','black'], numLines=10,title="Iterative prediction and correction from t=0 to t=T ({}) ".format(pm.ModelPath))
        print('iterative done')
        print('loading figures to screen')
        fig_stepWise.show()
        fig_stepWiseDist_M_N.show()
        fig_stepWiseDist_C_N.show()
        fig_stepWiseDist_C_M.show()
        fig_stepWiseDist_DistDiff.show()
        fig_iterative.show() 
# %%
