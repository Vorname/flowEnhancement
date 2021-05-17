
#%%
# %%
#tmp = data["architecture","units","error"]
#%%
# contains functions used by visualisation
# does NOT depend on parameter.py

import numpy as np
import sys
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def plotCoeffs(flowList, M, N, colors, singleScale=True, numLines=10, plotMaxima = True, ephasizeBaseFlow = True, highLightMean=False, title = 'Empty title', plotFull=False):
    # TODO: fix, every flow should have its own min and max
    # compute min and max omega from flowList
    w_min = np.inf
    w_max = -np.inf
    for flow in flowList:
        # TODO: only take basis < M in account for min and max
        w_min_tmp = np.min(flow)
        w_max_tmp = np.max(flow)
        w_min = w_min if w_min < w_min_tmp else w_min_tmp
        w_max = w_max if w_max > w_max_tmp else w_max_tmp

    x_axis_data = [i for i in range(np.shape(flowList[0])[-1])]
    
    visible_coeffs = M
    if plotFull:
        visible_coeffs = N

    # axis can be shared but this makes zooming impractical
    fig = make_subplots(visible_coeffs, visible_coeffs, False, False)
    numFlow = -1
    for flow in flowList:
        numFlow = numFlow+1
        flowMean = np.mean(flow, axis=0)
        for i in range(1,visible_coeffs+1):
            for j in range(1,visible_coeffs+1):
                # draw supset of training Data
                for k in range(0, np.shape(flow)[0], int(np.ceil(np.shape(flow)[0]/numLines))): 
                    fig.append_trace(go.Scatter(
                        x=x_axis_data[:],
                        y=flow[k,(i)*(N+1)+j,:],
                        #yaxis = 'y',
                        mode='lines',
                        opacity=1 if k == 0 and ephasizeBaseFlow else 1/numLines,
                        line=dict(color=colors[numFlow]), 
                        fill = 'none',
                        showlegend=False
                    ), row=i, col=j)
                    if singleScale:
                        fig.append_trace(go.Scatter(
                            x=[0,0],
                            y=[w_min,w_max],
                            opacity=0
                        ), row=i,col=j)
                if plotMaxima:
                    # plot min and max bands in .8 opacity
                    fig.append_trace(go.Scatter(
                        x=x_axis_data[:],
                        y=np.max(flow,axis=0)[(i)*(N+1)+j,:],
                        #yaxis = 'y',
                        mode='lines',
                        opacity=0.8,
                        line=dict(dash='dash',color=colors[numFlow]), 
                        fill = 'none',
                        showlegend=False
                    ), row=i, col=j)
                    fig.append_trace(go.Scatter(
                        x=x_axis_data[:],
                        y=np.min(flow,axis=0)[(i)*(N+1)+j,:],
                        #yaxis = 'y',
                        mode='lines',
                        opacity=0.8,
                        line=dict(dash='dash',color=colors[numFlow]), 
                        fill = 'none',
                        showlegend=False
                    ), row=i, col=j)
                if highLightMean:
                    fig.append_trace(go.Scatter(
                        x=x_axis_data[:],
                        y=flowMean[i*(N+1)+j,:],
                        #yaxis = 'y',
                        mode='lines',
                        opacity=1,
                        line=dict(color=colors[numFlow]), 
                        fill = 'none',
                        showlegend=False
                    ), row=i, col=j)
    fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    fig.update_layout(
    title=title,
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    ))
    fig.update_xaxes(
        gridcolor = "lightgray",
        zerolinecolor ="lightgray"
        )
    fig.update_yaxes(
        gridcolor = "lightgray",
        zerolinecolor = "lightgray"
    )
    return fig


def plotDistance(flowList, colorList, stopAt=None, numLines=10, stepOffset=0, plotMinMeanMax=True, title = 'Empyt title'):
        
        x_axis_data = [i for i in range(np.shape(flowList[0])[-1]-stepOffset)]
        # axis can be shared but this makes zooming impractical
        fig = go.Figure()

        if stopAt is None:
            stopAt = np.shape(flowList[0])[-1]

        numFlow = -1
        for flow in flowList:
            numFlow = numFlow + 1
            # compute min and max omega from flowList
            d_min = np.min(flow,axis=0)
            d_max = np.max(flow,axis=0)
            d_mean = np.mean(flow,axis=0)
            d_70 = np.percentile(flow,25,axis=0)
            d_30 = np.percentile(flow,75,axis=0)
            
            # TodoY dont print every line
            indxK = np.round(np.linspace(0,np.shape(flow)[0]-1,numLines)).astype(int)
            for k in indxK:
                fig.add_trace(go.Scatter(
                    x = x_axis_data,
                    y = flow[k,stepOffset:stopAt],
                    mode='lines',
                    fill='none',
                    opacity=1/numLines,
                    line=dict(color=colorList[numFlow]),
                    showlegend=False
                ))
            # Plot min, max, mean, p30, p70 seperately
            if plotMinMeanMax:
                fig.add_trace(go.Scatter(
                    x = x_axis_data,
                    y = d_min[stepOffset:stopAt],
                    mode='lines',
                    fill='none',
                    opacity=0.8,
                    line=dict(dash='dash',color=colorList[numFlow]),
                    showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x = x_axis_data,
                    y = d_max[stepOffset:stopAt],
                    mode='lines',
                    fill='none',
                    opacity=0.8,
                    line=dict(dash='dash',color=colorList[numFlow]),
                    showlegend=False
                ))
                # add the mean flow as solid line
                fig.add_trace(go.Scatter(
                    x = x_axis_data,
                    y = d_mean[stepOffset:stopAt],
                    mode='lines',
                    fill='none',
                    opacity=1,
                    line=dict(color=colorList[numFlow]),
                    showlegend=False
                ))
                # add the .3 percentil 'flow' as solid line
                fig.add_trace(go.Scatter(
                    x = x_axis_data,
                    y = d_30[stepOffset:stopAt],
                    mode='lines',
                    fill='none',
                    opacity=1,
                    line=dict(dash='dot',color=colorList[numFlow]),
                    showlegend=False
                ))
                # add the .7 percentil 'flow' as solid line
                fig.add_trace(go.Scatter(
                    x = x_axis_data,
                    y = d_70[stepOffset:stopAt],
                    mode='lines',
                    fill='none',
                    opacity=1,
                    line=dict(dash='dot',color=colorList[numFlow]),
                    showlegend=False
                ))
        fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        fig.update_layout(
        title=title,
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        ))
        fig.update_xaxes(
            gridcolor = "lightgray",
            zerolinecolor ="lightgray"
        )
        fig.update_yaxes(
            gridcolor = "lightgray",
            zerolinecolor = "lightgray"
    )
        return fig

def plotBandEnergy(flowList, colorList, valSamples, M, N, T, numLines = 10, showDifference=False, title='empty'):
    # compute band energy for every flow in flowList:
    K = np.shape(flowList[0])[0] # get the number of distinct flows in each flow group
    dimX = 5
    dimY = 4
    # TODO: dims should be dynamic
    x_axis_data = [i for i in range(T)]

    energyList = []
    # compute energy: needs rework, vectorization!
    for flow in flowList:
        energy = np.zeros((K,T,M))
        flow = np.reshape(np.swapaxes(flow,1,2),(valSamples, T, N+1, N+1)) # create a view to ease indexing
        for l in range(1,M+1):
            # flow shape: [pm.val_samples, (pm.basis_N+1)**2, pm.creational_steps] : coeffs larger M are zero 
            # l1 is simply flow_1,1 **2
            # lx = E(w_x)-E(w_x-1), with E(w_x-1)=sum^x-1(l_x)
            #energy[:,l-1,:] = np.sum(flow[:,])
            energy[:,:,l-1] = np.sum(flow[:,:,:l,:l]**2,(-1,-2)) # compute energy for the subspace <l
            if l > 1: # sqrt above?
                energy[:,:,l-1] -= np.sum(energy[:,:,:l-1],-1) # account  energy in subspace <l-1
        energy = np.swapaxes(energy,1,2) # unnecessary if code below gets changed accordingly
        energyList.append(np.copy(energy))


    fig = make_subplots(dimX, dimY, False, False)
    lin_idx = 0
    flow_idx = 0

    for energy in energyList:
        # Difference to itself is zero
        if showDifference and flow_idx==0:
            flow_idx = 1
            continue
        for k in range(K):
            lin_idx = 0
            for i in range(dimX):
                for j in range(dimY):
                    fig.append_trace(go.Scatter(
                        x=x_axis_data,
                        y=energy[k,lin_idx,:]-(energyList[0])[k,lin_idx,:]  if showDifference else energy[k,lin_idx,:],
                        #yaxis = 'y',
                        mode='lines',
                        opacity=1/K,
                        line=dict(color=colorList[flow_idx]), 
                        fill = 'none',
                        showlegend=False
                        ), row=i+1, col=j+1)
                    lin_idx = lin_idx + 1
        flow_idx = flow_idx + 1
    fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    fig.update_layout(
        title=title,
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        ))
    fig.update_xaxes(
        gridcolor = "lightgray",
        zerolinecolor ="lightgray"
        )
    fig.update_yaxes(
        gridcolor = "lightgray",
        zerolinecolor = "lightgray"
    )
    return fig

def plotBandEnergyDistance(flowList, colorList, valSamples, M, N, T, numLines = 10, useLog = False, normalize = False, axisEqual=False, plotDist=False, stopAt=None, title='empty'):
    # compute band energy for every flow in flowList:
    # plotDiff expects exact 3 flows: [N, M, C]
    K = np.shape(flowList[0])[0] # get the number of distinct flows in each flow group
    # dim is fixed for M=20
    dimX = 5
    dimY = 4

    # TODO: dims should be dynamic
    x_axis_data = [i for i in range(T)]

    if stopAt is None:
        stopAt = T

    numL = M
    baseFlowView = np.reshape(np.swapaxes((flowList[0]),1,2),(K,T,N+1,N+1))

    # switch computation by reshaping data to [K,T,M,M] and cmputing...
    energyList = []
    for flow in flowList: #this appends 0 for the distance to itself (zero flow) could be improved, but is already ommitted below
        energy = np.zeros((K+1,numL,T)) # there are M energy bands in a flow with M**2 coeffs
        #reshape flow to make computations simpler expect flow in shape: [K,(N+1)**2,T]
        print(np.shape(np.swapaxes(flow,1,2)))
        flowView = np.reshape(np.swapaxes(flow,1,2),(K,T,N+1,N+1))
        print(np.shape(flowView))
        for k in range(K):
            for l in range(1,numL+1):
                energy[k,l-1,:] = np.copy((np.sum((flowView[k,:,:l+1,:l+1])**2,axis=(-2,-1))-np.sum((flowView[k,:,:l,:l])**2,axis=(-2,-1))) - (np.sum((baseFlowView[k,:,:l+1,:l+1])**2,axis=(-2,-1))-np.sum((baseFlowView[k,:,:l,:l])**2,axis=(-2,-1))))
                if normalize:
                    energy[k,l-1,:] = energy[k,l-1,:]/(2*l-1)
        energy[K,:,:] = np.mean(energy[:-1,:,:],axis=0)
        if useLog:
            energyList.append(np.copy(np.log(energy+1)))
        else:
            energyList.append(np.copy(energy))

    print('energy bands computed')

    if plotDist:
        exit()
    
    # compute mean energy
            

    if axisEqual:
        minE = np.Inf
        maxE = np.NINF
        for e in energyList:
            minE = min(minE, np.min(e[:,:,:stopAt].flatten()))
            maxE = max(maxE, np.max(e[:,:,:stopAt].flatten()))

    print('start energy band plotting')
    fig = make_subplots(dimX, dimY, False, False)
    lin_idx = 0
    flow_idx = 0

    for energy in energyList:
        # Skip first flow since distance to itself is always zero
        if flow_idx == 0:
            flow_idx = flow_idx + 1
            continue
        if plotDist:
            energy = energyList[3]
            flow_idx = 0
        for k in range(0,K+1,5):
            lin_idx = 0
            for i in range(dimX):
                for j in range(dimY):
                    fig.append_trace(go.Scatter(
                        x=x_axis_data[:stopAt],
                        y=energy[k,lin_idx,:stopAt],
                        #yaxis = 'y',
                        mode='lines',
                        opacity=0.06 if k<K else 0.8,
                        line=dict(color=colorList[flow_idx]), 
                        fill = 'none',
                        showlegend=False
                        ), row=i+1, col=j+1)
                    # force equal axis along all plots
                    if axisEqual:
                        fig.append_trace(go.Scatter(
                            x=[0,0],
                            y=[minE,maxE],
                            opacity=0
                        ), row=i+1, col=j+1)

                    lin_idx = lin_idx + 1
        flow_idx = flow_idx + 1
        if plotDist:
            break
    fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", autosize=False, width=2000, height=1000,)
    fig.update_layout(
        title=title+" <br> Log: {}, normalize: {}, axisEqual: {}, diff: {}".format(useLog, normalize, axisEqual, plotDist),
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        ))
    fig.update_xaxes(
        gridcolor = "lightgray",
        zerolinecolor ="lightgray"
        )
    fig.update_yaxes(
        gridcolor = "lightgray",
        zerolinecolor = "lightgray"
    )
    return fig


def plotEnergy(flowList, colorList, valSamples, M, N, T, stopAt=None, normalize=True):
    # compute band energy for every flow in flowList:
    # plotDiff expects exact 3 flows: [N, M, C]
    K = np.shape(flowList[0])[0] # get the number of distinct flows in each flow group
    print(np.shape(flowList))
    print(np.shape(flowList[0]))

    numL = M

    if stopAt is None:
        stopAt = T

    x_axis_data = [i for i in range(stopAt)]
    
    # switch computation by reshaping data to [K,T,M,M] and cmputing...
    energyList = []
    for flow in flowList: #this appends 0 for the distance to itself (zero flow) could be improved, but is already ommitted below
        energy = np.zeros((K,numL,T)) # there are M energy bands in a flow with M**2 coeffs
        #reshape flow to make computations simpler expect flow in shape: [K,(N+1)**2,T]
        print(np.shape(np.swapaxes(flow,1,2)))
        flowView = np.reshape(np.swapaxes(flow,1,2),(K,T,N+1,N+1))
        print(np.shape(flowView))
        for k in range(K):
            for l in range(1,numL+1):
                energy[k,l-1,:] = np.copy(np.sum((flowView[k,:,:l+1,:l+1])**2,axis=(-2,-1))-np.sum((flowView[k,:,:l,:l])**2,axis=(-2,-1)))
                if normalize:
                    energy[k,l-1,:] = energy[k,l-1,:]/(2*l-1)
        energyList.append(np.copy(energy))

    print('energy bands computed')

    singleFlow = (flowList[0])[0,:,:]
    print(np.shape(singleFlow))

    """     fig = go.Figure()
    num_energy = -1
    for flow in flowList:
        num_energy = num_energy+1
        fig.add_trace(go.Scatter(
            x = x_axis_data,
            y = np.sum(flow[0,:,:]**2,axis=0),
            mode='lines',
            fill='none',
            line=dict(color=colorList[num_energy])
        )) """

    fig = go.Figure()
    num_energy = -1
    for energy in energyList:
        num_energy = num_energy+1
        fig.add_trace(go.Scatter(
            x = x_axis_data,
            y = np.sum(energy[0,:,:],axis=0),
            mode='lines',
            fill='none',
            line=dict(color=colorList[num_energy])
        ))

    fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(
        gridcolor = "lightgray",
        zerolinecolor ="lightgray"
        )
    fig.update_yaxes(
        gridcolor = "lightgray",
        zerolinecolor = "lightgray"
    )

    return fig
