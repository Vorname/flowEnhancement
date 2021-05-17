#%%
import numpy as np
import scipy as sp
from integration import *
from global_functions import *
import os
from flow_data import FlowData
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from Log import MatrixToString

from bandwithComparison import BandwithData
from render import RenderDirect

from parameter import *
from reconstruction import *

import lic
import png
import matplotlib.pyplot as plt # carefull with plotly, crashes often one another




#%%

if __name__ == "__main__":

    # dis has not been used in a long time but contains several nice functions for training flow analysis and stuff
    # probably broken
    
    num_iterations = 1
    useEuler = True

    # parameters set in parameter.py
    num_basis_M = 2
    num_basis_N_vec = [3, 6, 12, 20, 32, 48]
    num_basis_initial = 2 # must be 0 < var <= M

    forceAxisEqual = False
    num_rows_plt = basis_M+1
    num_cols_plt = basis_M+1
    ImageDim = 1024
    #snapPoints = [0*fps*2,1*fps*2,2*fps*2,3*fps*2,4*fps*2,5*fps*2,6*fps*2,7*fps*2,8*fps*2-1]# snap a picture every second of integration
    snapPoints = [2*fps*2, 4*fps*2, 6*fps*2]

    printN = True
    numDots = 500
    numXticks = 5
    viscosity = visc 
    t_end = creational_steps*4 # times 2 is working with default fps
    fps = fps*2 # 128 is not enough, numXticks takes carce of conversion to seconds

    print("T={}sec".format(t_end/fps))
    if num_iterations == 1:
        epsilon = 0 # 0 for same flow in all iterations
    else:
        epsilon = eps
    use_fft = True

#%%
    # adapt second parameter in format to allow unspecified flow flie
    initialFlow = np.load("./initialFlows/M{}_N{}_baseFlow.npy".format(num_basis_M, num_basis_N_vec[0]))# load seed
    num_N_largest = num_basis_N_vec[-1]
    # make this size+1 for the corrected output via model
    N_data = np.zeros((len(num_basis_N_vec)+1, t_end, num_iterations, num_N_largest + 1, num_N_largest + 1))

    # load N_data if working on the default set:
    if os.path.exists("../data/M{}_N{}_simAnalys.npy".format(num_basis_M, num_basis_N_vec)):
        N_data = np.load("../data/M{}_N{}_simAnalys.npy".format(num_basis_M, num_basis_N_vec))
        print("data loaded: skipping computation!")
    else:
        exit()
        # compute flows
        for i, num_basis_N in enumerate(num_basis_N_vec):
            print('doing N={} with i={}'.format(num_basis_N, i))
            initialFlowData = np.zeros(((num_basis_N+1)**2, num_iterations))
            for v in range(num_iterations):
                tmpInitialFlow = np.zeros((num_basis_N+1, num_basis_N+1))
                tmpInitialFlow[:num_basis_N_vec[0]+1, :num_basis_N_vec[0]+1] = initialFlow
                initialFlowData[:,v] = tmpInitialFlow.flatten()

            # special print
            if num_basis_M == 2:
                print(initialFlow[1:3,1:3])

            print(np.shape(initialFlow))
            
            test_data = BandwithData(num_basis_M, num_basis_N, t_end, fps, num_iterations, initialFlowData=initialFlowData, viscosity=viscosity, epsilon=epsilon, useEuler=useEuler)

            print('starting data computation\n')
            test_data.ComputeData(normalizeEpsilonFlow=False, use_fft=use_fft)
            print('computation finished\n')

            # np.zeros([(basis_N+1)**2, int(t_end), num_iterations])
            tmpData = np.copy(test_data.get_N())
            tmpData = np.moveaxis(tmpData, 0,-1)
            print("tmpShape: {} at i={}".format(np.shape(tmpData), i))
            tmpData = np.reshape(tmpData, (t_end, num_iterations, num_basis_N+1, num_basis_N+1), order = "C") # order?
            N_data[i+1, :, :, :num_basis_N+1, :num_basis_N+1] = np.copy(tmpData)


            # store m once in the n stroage as well
            if i==0:
                tmpData = np.copy(test_data.get_M())
                tmpData = np.moveaxis(tmpData, 0,-1)
                tmpData = np.reshape(tmpData, (t_end, num_iterations, num_basis_M+1, num_basis_M+1), order = "C") # order?
                N_data[0, :, :, :num_basis_M+1, :num_basis_M+1] = np.copy(tmpData)

#%%
    # use lic and reconstruction to create flow image
    print("start plotting lic images")
    for i in range(len(num_basis_N_vec)+1):
        num_basis = 2 if i==0 else num_basis_N_vec[i-1]
        for snapPoint in snapPoints:
            res =Reconstruct_Velocity(N_data[i,int(snapPoint),0,:,:].flatten(), None, True, res=(ImageDim,ImageDim))
            lic_result = lic.lic(res[1,:,:],res[0,:,:])# somehow x and y are switched in res
            png.from_array((lic_result*255).astype('uint8'), 'L').save("./LICImages/FlowLIC_T{}_N{}.png".format(int(snapPoint)/fps, num_basis))
            #ffmpeg -ss 30 -t 3 -i input.mp4 -vf "fps=10,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 output.gif




#%% 
    # Plot the N_store straight here
    #shape: N_data = np.zeros((len(num_basis_N_vec)+1, num_N_largest + 1, num_N_largest + 1, t_end, num_iterations))

    print("\n start printing \n")
    N_idx = np.round(np.linspace(0, t_end - 1, numDots)).astype(int)
    x_axis_data = [i for i in range(int(len(N_idx)))]
    x_axis_ticks = np.round(np.linspace(0, t_end - 1, numXticks)).astype(int)
    colorArray = ["black", "#5DC9F0", "#658FF7", "#7B67E0", "#C265F7", "#ED61D6", "#ED6190"] # made with adobe oclorpicker

    combifig = make_subplots(rows=num_rows_plt , cols=num_cols_plt, start_cell="top-left")#,shared_xaxes=share_x,shared_yaxes=share_y)

    # force axis to be shared row wise using fake data points
    min_val = np.amin(N_data)
    max_val = np.amax(N_data)
    max_x = 0#np.amax(x_axis_data)
    
    for num_Flow in range(len(num_basis_N_vec)+1):
        for slice_index in range(0, num_iterations):
            for i in range(1, num_rows_plt + 1):
                for j in range(1, num_cols_plt + 1):
                    combifig.append_trace(go.Scatter(
                    x=x_axis_data,
                    y=N_data[num_Flow, N_idx, slice_index, i, j],
                    yaxis = 'y',
                    mode='lines',
                    opacity=0.8,
                    line=dict(color=colorArray[num_Flow], width=4),
                    fill = 'none',
                    name = "N = {}".format(np.insert(num_basis_N_vec, 0, num_basis_M, axis=0)[num_Flow])
                    ), row=i, col=j)

                    # set min and max in last column for every subplot to force equal axis along rows
                    if (forceAxisEqual):
                        combifig.append_trace(go.Scatter(
                        x = [max_x],
                        y = [min_val],
                        mode='markers',
                        opacity=0,
                        ), row=i, col=j)
                        combifig.append_trace(go.Scatter(
                        x = [max_x],
                        y = [max_val],
                        mode='markers',
                        opacity=0,
                        ), row=i, col=j)  
    combifig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    combifig.update_xaxes(
            tickmode = 'array',
            tickvals = np.round(np.linspace(0, len(N_idx) - 1, numXticks)).astype(int), # create 10 x-ticks
            ticktext = np.round(x_axis_ticks/fps,2),
            gridcolor = "lightgray",
            zerolinecolor ="lightgray"
        )
    combifig.update_yaxes(
        gridcolor = "lightgray",
        zerolinecolor = "lightgray"
    )
    #combifig.update_yaxes(title_text="K2", row=1, col=1)
    combifig.show()
    print('start writing image')
    #combifig.write_image("../../writing/thesis/diagrams/M{}_N{}_x{}.pdf".format(num_basis_M, num_basis_N_vec, num_rows_plt))
    combifig.write_image("../../writing/thesis/diagrams/COEF_M{}_N{}_x{}_eq{}.svg".format(num_basis_M, num_basis_N_vec, num_rows_plt, forceAxisEqual))
    print('imag 1 written')

#%% compute and plot energy here:
    #N_data : np.zeros((len(num_basis_N_vec)+1, t_end, num_iterations, num_N_largest + 1, num_N_largest + 1))
    N_energy = np.sum(N_data**2,axis=(-1,-2)) # compute energy (uses parcevals identity for rothonormal laplacain filesd) # use all the available or only M ones?
    #N_energy = N_energy[:,:,:] - N_energy[0,:,:] # normalize for first axis (correct?)
    energy_fig = go.Figure()
    for num_Flow in range(len(num_basis_N_vec)+1):
        for slice_index in range(0, num_iterations):
            energy_fig.add_trace(go.Scatter(
                    x=x_axis_data,
                    y=N_energy[num_Flow, N_idx, slice_index],
                    yaxis = 'y',
                    mode='lines',
                    line=dict(color=colorArray[num_Flow], width=4),
                    fill = 'none',
                    name = "N = {}".format(np.insert(num_basis_N_vec, 0, num_basis_M, axis=0)[num_Flow])
                    ))
    energy_fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")

    energy_fig.update_xaxes(
            tickmode = 'array',
            tickvals = np.round(np.linspace(0, len(N_idx) - 1, numXticks)).astype(int), # create 10 x-ticks
            ticktext = np.round(x_axis_ticks/fps,2),
            gridcolor = "lightgray",
            zerolinecolor ="lightgray"
        )
    energy_fig.update_yaxes(
        gridcolor = "lightgray",
        zerolinecolor = "lightgray"
    )

    energy_fig.show()
    print('start writing image')
    #combifig.write_image("../../writing/thesis/diagrams/M{}_N{}_x{}.pdf".format(num_basis_M, num_basis_N_vec, num_rows_plt))
    energy_fig.write_image("../../writing/thesis/diagrams/ENG_M{}_N{}_x{}.svg".format(num_basis_M, num_basis_N_vec, num_rows_plt))
    print('image 2 written')


#%%
    exit()
    print('plotting... \n')
    #p_dist = test_data.PlotDistance()
    p_multi_M, p_multi_N = test_data.PlotMultipleSlice(forceAxisEqual=False, filterZero=True,share_y=False)
    p_multi_M.show()
    #p_multi_N.show()
    #p_multi_f.show()
    #p_dist.show()
    test_data = None
    p_multi_M = None
    p_multi_N = None