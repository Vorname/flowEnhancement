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



# exclude this in seperate file:
###
def matrix2LLinear(A):
    assert np.shape(A)[0]==np.shape(A)[1], 'matrix must be square'
    res = np.zeros((np.shape(A)[0]**2))
    for i in range(0,np.shape(A)[0]):
        for j in range(0,np.shape(A)[0]):
            # Compute this fucking index once and for all..
            wave = max(i,j)
            prev_elements = wave**2
            if i < j:
                k = i
            else:
                k = j + wave
            res[prev_elements+k] = A[i,j]
    return res

def LLinear2matrix(A):
    M = int(np.sqrt(len(A)))
    res = np.zeros((M,M))
    for i in range(M):
        for j in range(M):
            wave = max(i,j)
            prev_elements = wave**2
            if i < j:
                k = i +1
            else:
                k = j + wave +1
            res[i,j] = A[prev_elements+k]
    return res

def plotKSlice(A,k_slice, title = 'none'):
    assert np.shape(A)[0]==np.shape(A)[1], 'matrix must be square'
    fig = go.Figure(data=go.Heatmap(
                    z=A,
                    colorscale='Picnic',
                    xgap =	3,
                    ygap =	3)
                    )
    fig.update_layout(
                    title=title,
                    autosize=False,
                    width=1400,
                    height=1000,
                    yaxis=dict(
                        scaleanchor='x',
                        scaleratio=1,
                        autorange='reversed'))
    return fig
###

##################################################################

## parameters
num_basis_M = 4
k1 = 3
k2 = 3
k_slice = (k1-1)*num_basis_M+(k2-1)

M_sqrt = num_basis_M**2

adv_M_full = BuildAdvectionMatrix(num_basis_M, useFullShape=True)
adv_M = BuildAdvectionMatrix(num_basis_M)


A = np.zeros((num_basis_M,num_basis_M,num_basis_M,num_basis_M,num_basis_M,num_basis_M))
A_Lin_conventionel = np.zeros((num_basis_M**2,num_basis_M**2,num_basis_M**2))


# Create full matrix from sparse and remove zero entries from fft
#    # Linearize matrix
Lin_A = np.zeros((num_basis_M**2,num_basis_M**2,num_basis_M**2))
for k1 in range(num_basis_M):
    for k2 in range(num_basis_M):
        for i1 in range(num_basis_M):
            for i2 in range(num_basis_M):
                for j1 in range(num_basis_M):
                    for j2 in range(num_basis_M):
                        A[k1,k2,i1,i2,j1,j2] = adv_M_full[k1+1,k2+1,i1+1,i2+1,j1+1,j2+1]
                        A_Lin_conventionel[k1*num_basis_M+k2,i1*num_basis_M+i2,j1*num_basis_M+j2] = adv_M[k1+1,k2+1,i1+1,i2+1,j1+1,j2+1]
#linearize j
Lin_tmp_j = np.zeros((num_basis_M,num_basis_M,num_basis_M,num_basis_M,num_basis_M**2))
for k1 in range(num_basis_M):
    for k2 in range(num_basis_M):
        for i1 in range(num_basis_M):
            for i2 in range(num_basis_M):
                Lin_tmp_j[k1,k2,i1,i2,:] = matrix2LLinear(A[k1,k2,i1,i2,:,:])

# linearize i
Lin_tmp_ij = np.zeros((num_basis_M,num_basis_M,num_basis_M**2,num_basis_M**2))
for k1 in range(num_basis_M):
    for k2 in range(num_basis_M):
        for j in range(num_basis_M**2):
            Lin_tmp_ij[k1,k2,:,j] = matrix2LLinear(Lin_tmp_j[k1,k2,:,:,j])

# linearize k
#Lin_A = np.zeros((num_basis_M**2,num_basis_M**2,num_basis_M**2))
#for i in range(num_basis_M**2):
#    for j in range(num_basis_M**2):
#        Lin_A[:,i,j] = matrix2LLinear(Lin_tmp_ij[:,:,i,j])


# Embed matrices into one big
B = np.zeros((num_basis_M**3,num_basis_M**3))
B_conv = np.zeros((num_basis_M**3,num_basis_M**3))

for i in range(num_basis_M):
    for j in range(num_basis_M):
        B[i*M_sqrt:(i+1)*M_sqrt, j*M_sqrt:(j+1)*M_sqrt] = Lin_tmp_ij[i,j,:,:]
        B_conv[i*M_sqrt:(i+1)*M_sqrt, j*M_sqrt:(j+1)*M_sqrt] = A_Lin_conventionel[i*num_basis_M+j,:,:]


fig = plotKSlice(B,num_basis_M, 'hierarchical with M = {}'.format(num_basis_M))
fig.show()
fig.write_image("./hierarchical with M = {}.png".format(num_basis_M))

fig = plotKSlice(B_conv,num_basis_M,'conventional with M = {}'.format(num_basis_M))
fig.show()
fig.write_image("./conventional_M_{}.png".format(num_basis_M))

print(k_slice)
fig = plotKSlice(A_Lin_conventionel[k_slice,:,:],num_basis_M,'slingle slice with k={}'.format(k_slice))
fig.show()
fig.write_image("./slice_k_{}_M_{}.png".format(k_slice,num_basis_M))

fig = plotKSlice(Lin_tmp_ij[k1,k2,:,:],num_basis_M,'slingle slice with k={}'.format(k_slice))
fig.show()
fig.write_image("./hierarchical_slice_k_{}_M_{}.png".format(k_slice,num_basis_M))

#%%