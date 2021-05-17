#%%
# global imports:
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import time

# local imports:
from integration import *
import parameter as pm

#%%
if __name__ == "__main__":
        
    # This skript compares a number of fows in terms of its enery flows into low/high frequencies.
    # Use to create a inital flow suitable for train data generation.   

    ## set parameters
    num_basis_M = pm.basis_M
    num_basis_N = pm.basis_N
    num_tests = 1000 # >8 for drawing
    # store the best result for use as training data here:
    fileName = './initialFlows/M{}_N{}_baseFlow.npy'.format(num_basis_M, num_basis_N)

    # initialize flow at t0 using M or N bases
    full_N = False
    # if set to Flase only basis <M will be none-zero and the flow into N will be used as a score.
    # Otherwise the over all flow into M, which depends on the initialization of N, will be used as score.
    # The latter one produces an initially more turbulent but overall more stable seed an is prefered.

    M_sqrt = num_basis_M**2
    N_sqrt = num_basis_N**2

    # set logging matrices
    w_store = np.zeros((num_tests,num_basis_N+1,num_basis_N+1))
    w_dot_store = np.zeros((num_tests,num_basis_N+1,num_basis_N+1))
    flowToM = np.zeros((num_tests))
    flowToN = np.zeros((num_tests))
    energy = np.zeros((num_tests))

    print('start computation \n')
    for i in range(num_tests):
        startTime = time.time()
        # must include +1 for fft (k=0)
        w = np.zeros((num_basis_N+1, num_basis_N+1))
        
        # Initialize w_K<N or w_k<M using uniform distribution in [-1,1)
        if full_N:
            # Initialize the whole bandwidth
            w[1:,1:] =  (np.random.random_sample((num_basis_N,num_basis_N))-0.5)*2
        else:
            # Initialize only up to M basis, this twerkadoodels everything, only use if you know what you are doing
            w[1:num_basis_M+1,1:num_basis_M+1] =  (np.random.random_sample((num_basis_M,num_basis_M))-0.5)*2
            
        # normalization optional: 
        #w = w/np.sum(np.sqrt(abs(w)))

        w = np.reshape(w,(num_basis_N+1)**2)

        # Compute w_dot and flow into M, N respectively
        w_dot, _ = Advection(w,None,0,usefft=True)
        w_dot = np.reshape(w_dot,(num_basis_N+1,num_basis_N+1))
        w = np.reshape(w,(num_basis_N+1,num_basis_N+1))
        flowToM[i] = np.sum(w_dot[:num_basis_M+1,:num_basis_M+1]**2) # energy Transport into M, delta into M
        flowToN[i] = np.sum(w_dot**2)-flowToM[i] # energy Transport into N, delta into N
        energy[i] = np.sum(w**2) # "energy" in N
        w_store[i,:,:] = w
        w_dot_store[i,:,:] = w_dot
        if np.mod(i,num_tests/10)==0:
            print("{}/{} done in {} sec".format(i,num_tests,time.time()-startTime))
    # optimize w_dot for w[k<M] and w_dot[k>M]
    print('finished computation \n')

    #%%

    # visualize quality

    # get sorting indices
    if full_N:
        idx = np.argsort(flowToM)
    else:
        idx = np.argsort(flowToN)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=energy[idx], name = 'M'))
    fig.add_trace(go.Scatter(y=flowToN[idx], name = 'flow to N'))
    fig.add_trace(go.Scatter(y=flowToM[idx], name = 'flow to M'))
    fig.show()

    # use a scatter plot for pareto optimal front
    fig = go.Figure(data=go.Scatter(x=energy[idx], y=flowToN[idx], mode='markers'))
    fig.update_layout(
    title="Flow into N",
    xaxis_title="Total Energy",
    yaxis_title="Energy flow into N",
    )
    fig.show()

    fig = go.Figure(data=go.Scatter(x=energy[idx], y=flowToM[idx], mode='markers'))
    fig.update_layout(
    title="Flow into M",
    xaxis_title="Total Energy",
    yaxis_title="Energy flow into N",
    )
    fig.show()

    fig = go.Figure(data=go.Scatter(x=flowToM[idx], y=flowToN[idx], mode='markers'))
    fig.update_layout(
    title="Flow divergence",
    xaxis_title="Energy flow into M",
    yaxis_title="Energy flow into N",
    )
    fig.show()

    # get top nine best (works only if all N basis are initialized none-zero)
    if full_N:
        idx = flowToM.argsort()[-9:][::-1]
    else:
        # use only if you know what you are doing:
        idx = flowToN.argsort()[-9:][::-1]
    fig = make_subplots(rows=3, cols=3, start_cell="top-left",shared_xaxes=True,shared_yaxes=True)
    print(idx)
    
    for i in range(1,4):
        for j in range(1,4):
            fig.append_trace(go.Heatmap(
                        z=w_store[idx[(i-1)*3+(j-1)],:,:],
                        colorscale='Picnic'
                        ), row=i, col=j)
            fig.update_yaxes(
                            scaleanchor='x',
                            scaleratio=1,
                            autorange='reversed', row=i, col=j)

    fig.show()

    # save the very best
    np.save(fileName,w_store[idx[0]])


