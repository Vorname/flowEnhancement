#%%
import numpy as np
from integration import *
from global_functions import *
import os
from reconstruction import *
import sys
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import lic
import png
import matplotlib.pyplot as plt # carefull with plotly, crashes often one another

#%%

if __name__ == "__main__":

    # create individual coefficient flows with minimal ink

    pathToFlow = "./dummy_v{}.npy"
    T = 1
    fps = 128
    N = 12
    M = 6
    visible_coeffs = N
    visc = 0.01 # v = 0 distortes local correlation greatly. (run for T > 12 for better results)
    ImageDim = 512

    steps  = np.round(np.linspace(0,T*fps-1,24*T)).astype('int')
    points = np.round(np.linspace(0,T*fps-1,100)).astype('int')

    print(steps)
    print(len(steps))

    T0 = np.zeros((N+1,N+1))
    T0[1,1] = 1
    T0[1,2] = 1
    T0[2,1] = 1

    print(np.shape(T0))

    if not os.path.exists(pathToFlow.format(visc)):
        print("start integration..")    
        flow = np.zeros([(N+1)**2, int(T*fps)])
        IntegrateSimple(T0.flatten(), visc, int(T*fps), fps, None, coeff_data = flow)
        print("integration done")
        np.save(pathToFlow.format(visc),flow)
    else:
        print("loading")
        flow = np.load(pathToFlow.format(visc))

    # compute energy in M and N for comparison
    energyN = np.sum(flow**2,axis=0)
    energyM = 0
    for i in range(M+1):
        for j in range(M+1):
            energyM = energyM + flow[(i)*(N+1)+j,:]**2
    energyNonly = energyN - energyM + energyM[0]

    fig = make_subplots(visible_coeffs, visible_coeffs, False, False)
    for i in range(1,visible_coeffs):
        for j in range(1,visible_coeffs):
            fig.append_trace(go.Scatter(
            #x=x_axis_data[:],
            y=flow[(i)*(N+1)+j,points],
            #yaxis = 'y',
            mode='lines',
            line=dict(color="#ED6190"), 
            fill = 'none',
            showlegend=False
            ), row=i, col=j)
    fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(
            showgrid=False,
            zeroline=False,
            showticklabels=False
        )
    fig.update_yaxes(
            showgrid=False,
            zeroline=False,
            showticklabels=False
            )
    fig.show()
    print("all done")
# %%
