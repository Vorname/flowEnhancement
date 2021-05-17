#%%
import numpy as np
import lic
import png

from integration import *
from reconstruction import *






#%%

if __name__ == "__main__":

    # create random flow in N
    # integrate random M, N respectively and create LICs for gif
    # adapt or use interactive to plot train data

    T = 1 # time in seconds

    fps = 128
    M = 6
    N = 12
    visc = 0
    ImageDim = 512

    steps = np.round(np.linspace(0,T*fps-1,24*T))
    # create storage matrices
    NFlow = np.zeros([(N+1)**2, int(T*fps)])
    MFlow = np.zeros([(M+1)**2, int(T*fps)])
    # N and M flow can be loaded from training data/ validation data or similiar as well

    initalFlow = np.zeros((N+1,N+1))
    randomSeed = (np.random.rand(M,M)-.5)*2
    initalFlow[1:M+1,1:M+1] = randomSeed

#%%
   
    IntegrateSimple(initalFlow.flatten(), visc, int(T*fps), fps, None, coeff_data = NFlow)

    print('first done')

    MBaseSeed = np.copy(np.reshape(initalFlow, (N+1,N+1))[:M+1,:M+1])
    
    IntegrateSimple(MBaseSeed.flatten(), visc, int(T*fps), fps, None, coeff_data = MFlow)

    print('last done')

    # ffmpeg -r 32 -start_number_range 383 -f image2 -i "FlowLIC_T%05d.png" -vcodec libx264 -pix_fmt yuv420p FLOW.mp4

#%%
    i = 0
    for t in steps:
        res =Reconstruct_Velocity(NFlow[:,int(t)].flatten(), None, True, res=(ImageDim,ImageDim))
        lic_result = lic.lic(res[1,:,:],res[0,:,:])# somehow x and y are switched in res
        png.from_array((lic_result*255).astype('uint8'), 'L').save("./LICN/FlowLIC_T{:05d}.png".format(int(i)))
        i += 1

    i=0
    for t in steps:
        res =Reconstruct_Velocity(MFlow[:,int(t)].flatten(), None, True, res=(ImageDim,ImageDim))
        lic_result = lic.lic(res[1,:,:],res[0,:,:])# somehow x and y are switched in res
        png.from_array((lic_result*255).astype('uint8'), 'L').save("./LICM/FlowLIC_T{:05d}.png".format(int(i)))
        i += 1

    print("all done")