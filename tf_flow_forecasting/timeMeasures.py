# this file is solely for time measurments of a given network stored in ./trainingSets
# results are console only

# parameters are set from the parameter.py file

# prerequisits: complete set in ./trainingSet with available data
import tensorflow as tf
import numpy as np
import sys
from makeCustomModel import *
import multiprocessing as mp
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# force execution on cpu for a 'fair' time comparison

# local imports
sys.path.append('../') # add root code folder to search path for includes

# Import parameters
#import advection.parameter as pm

def extractM(flow, K, T, M, N):
    ret = np.copy(np.reshape(np.swapaxes(flow,1,2), (K, T, N+1, N+1)))
    ret[:,:,M+1:,:] = 0
    ret[:,:,:,M+1:] = 0
    ret = np.copy(np.swapaxes(np.reshape(ret, (K, T, (N+1)**2)),1,2))
    return ret

def conventionalLSTMModel():
    init_std = ini_weight
    #initializer = tf.zeros_initializer()
    initializer = tf.random_normal_initializer(
    mean=0, stddev=init_std, seed=None
    )
    #stddev = sqrt(2 / (fan_in + fan_out))#glorot
    #stddev = sqrt(scale / (fan_in+fan_out)/2) VarianceScaling scale = 4 for glorot behaviour?
    #print('stddev={}'.format(np.sqrt(init_std/((steps*M*M+steps*N*N)/2))))
    initializer = tf.keras.initializers.VarianceScaling(
        scale=init_std, mode='fan_avg'
    )

    inputs = tf.keras.Input(shape=(steps, M, M))
    # branch left
    x2 = inputs[:,steps-1,:,:]
    # branch right
    x1 = tf.keras.layers.Reshape((steps, M**2))(inputs)
    x1 = tf.keras.layers.Dense(internalDim, activation=tf.keras.activations.tanh, kernel_initializer=initializer)(x1) 
    x1 = tf.keras.layers.LSTM(internalDim, return_sequences=True, kernel_initializer=initializer)(x1) #expand to N**2
    x1 = tf.keras.layers.Dense(internalDim, kernel_initializer=initializer)(x1) 
    x1 = tf.keras.layers.LSTM(internalDim, return_sequences=True, kernel_initializer=initializer,recurrent_initializer=initializer)(x1)
    x1 = tf.keras.layers.Dense(internalDim, kernel_initializer=initializer)(x1)
    x1 = tf.keras.layers.LSTM(internalDim, kernel_initializer=initializer,recurrent_initializer=initializer)(x1) # remove time dim
    x1 = tf.keras.layers.Dense(internalDim, activation=tf.keras.activations.tanh, kernel_initializer=initializer)(x1) # shrink dim to N^2
    x1 = tf.keras.layers.Dense(M**2, kernel_initializer=initializer)(x1)
    x1 = tf.keras.layers.Reshape((M, M))(x1)
    # Skip connection ends here
    outputs = tf.keras.layers.Add()([x2,x1]) # this could make problems if x2 is completly wrong sliced 
    #outputs = x1
    #TODO: return x2 uncompromised
    #outputs = x2
    simple_lstm_model = tf.keras.Model(inputs=inputs, outputs=outputs, name="LSTM")

    #keras.utils.plot_model(simple_lstm_model, 'LSTM.png', show_shapes=True)
    # available loss and metric:
   # simple_lstm_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    simple_lstm_model.compile(optimizer=optimizer, loss=loss , metrics=metrics)

    return simple_lstm_model


if __name__ == "__main__":
    start_time = time.time()
    tf.keras.backend.clear_session()
    T = 128 # acts as samples must be steps>
    M = 2#pm.basis_M 
    N_max = 56#pm.basis_N
    N = 2
    steps = 8#pm.steps #input steps
    fps = 128#pm.fps
    ini_weight = 1e0#pm.ini_weight
    internalDim = N**2 #N**2 #N**2 is default behaviour  # might depend on M or N


    # store time in arrays
    N_time = np.zeros(N_max)
    C_time = np.zeros(N_max)
    Input_size = np.zeros(N_max)
    Internal_size = np.zeros(N_max)

    # run the network once to start tf on the gpu
    model = conventionalLSTMModel()
    nextInput = np.zeros((1, steps, M, M))
    nextInput = tf.data.Dataset.from_tensor_slices((nextInput, None))
    nextInput = nextInput.batch(1)
    tmp = np.zeros((N+1,N+1))
    tmp[1:M+1,1:M+1] = np.squeeze(model.predict(nextInput))
    tf.keras.backend.clear_session()
    

    for N in range(2,N_max):
        M = int(max(2,int(np.round((N/5)*3)))) #Not really needed..., replace by loop?
        internalDim = M**2 #N**2 #N**2 is default behaviour  # might depend on M or N
        # swap to M
        print("currently: N: {} M: {}".format(N, M))

        # cerate data to work on. can be anything. really
        w_M = np.zeros((M+1,M+1))
        w_M[1:,1:] = np.random.rand(M,M)
        w_N = np.random.rand(N+1,N+1)
        w_N[:M+1,:M+1] = np.copy(w_M)
        # reshape
        w_M = np.reshape(w_M,((M+1)**2))
        w_N = np.reshape(w_N,((N+1)**2))

        # hard coded model path!!!
        #model = tf.keras.models.load_model('./trainingSets/run1000/trainedModel')
        model = conventionalLSTMModel()


        tmp_M = np.zeros(((M+1)**2,T))
        tmp_N = np.zeros(((N+1)**2,T))
        print('starting N')
        start_N_t = time.time_ns()
        start_N_p = time.process_time_ns()
        start_N_c = time.perf_counter_ns()
        IntegrateSimple(w_N,0,T,fps,coeff_data=tmp_N)
        end_N_t = time.time_ns()
        end_N_p = time.process_time_ns()
        end_N_c = time.perf_counter_ns()
        # integrate and correct on M
        print('starting C')
        start_C_t = time.time_ns()
        start_C_p = time.process_time_ns()
        start_C_c = time.perf_counter_ns()
        val_iterative_C = np.copy(tmp_N)
        val_stepWise_M = np.copy(tmp_M) # this is incorrect! should be stepwise
        # reverence flow precomputed by dataGeneration
        countC = 0
        for t in range(steps-1, T):
            countC = countC+1
            nextInput = np.empty((1, steps, M, M))
            nextInput[0,:,:,:] = np.copy(np.reshape(np.transpose(val_iterative_C[:,t-steps+1:t+1]),(steps,N+1,N+1))[:,1:M+1,1:M+1])
            M_temp = val_stepWise_M[:,t]
            nextInput[0,-1,:,:] = np.copy(np.reshape(M_temp,(M+1,M+1))[1:M+1,1:M+1])
            nextInput = tf.data.Dataset.from_tensor_slices((nextInput, None))
            nextInput = nextInput.batch(1)
            # nextInput could be a complete dummy one

            # Correct previous integration step using lstm
            tmp = np.zeros((N+1,N+1))
            #print('prediction!')
            tmp[1:M+1,1:M+1] = np.squeeze(model.predict(nextInput))
            val_iterative_C[:, t] = np.copy(tmp.flatten())
            end_C_t = time.time_ns()
            end_C_p = time.process_time_ns()
            end_C_c = time.perf_counter_ns()
        print("countC = {}".format(countC))
        # compare timings  
        print("N={} N: {}".format(N,(end_N_c-start_N_c)/1000000000/T))
        #print("N={} C: {} time".format(N,(end_C_t-start_C_t)/1000000000/T))
        #print("N={} C: {} proc time".format(N,(end_C_p-start_C_p)/1000000000/T))
        print("N={} C: {} perf count".format(N,(end_C_c-start_C_c)/1000000000/T))
        #print((end_C_c-start_C_c)/1000000000)
        #print("N={} M+C: {}".format(N,((end_C_p-start_C_p)+(end_M_p-start_M_p))/1000000000/T))
        N_time[N] = ((end_N_c-start_N_c)/T)/1e+9 # convert to sec
        C_time[N] = ((end_C_c-start_C_c)/countC)/1e+9 # convert to sec
        Input_size[N] = M
        Internal_size[N] = internalDim
        tf.keras.backend.clear_session()

    print(N_time)
    print(C_time)
    print(C_time + N_time[Input_size.astype(int)])
    print(Input_size)
    print(Internal_size)

    end_time = time.time()
    print("TotalTime: {}".format(end_time-start_time))




    # get execution time of network without memcopy time?
    # -> hard to achive with tf. profiling might work but has to be set up separately and built into python tf code..

# Print the whole madeness:
x_axis_data = [i for i in range(int(N_max))]
colorArray = ["black", "#5DC9F0", "#658FF7", "#7B67E0", "#C265F7", "#ED61D6"] # made with adobe oclorpicker

time_fig = go.Figure()
# add computation time for network with N**2 internal size (in|output is (N/5)*3 rounded)
time_fig.add_trace(go.Scatter(
        x=x_axis_data,
        y=C_time,
        yaxis = 'y',
        mode='lines',
        line=dict(color="#658ff7"),
        fill = 'none'
        ))
# add total integration plus enhancement time
time_fig.add_trace(go.Scatter(
        x=x_axis_data,
        y=C_time + N_time[Input_size.astype(int)],
        yaxis = 'y',
        mode='lines',
        line=dict(color="#ed6190"),
        fill = 'none'
        ))
# add network input size at internal size N: needs better plotting
time_fig.add_trace(go.Scatter(
        x=x_axis_data,
        y=Internal_size/Internal_size[-1],
        yaxis = 'y',
        mode='lines',
        line=dict(color="lightgray"),
        line_shape='hv',
        fill = 'none'
        ))
# add raw integration time according to number of basis in use
time_fig.add_trace(go.Scatter(
        x=x_axis_data,
        y=N_time,
        yaxis = 'y',
        mode='lines',
        line=dict(color="black"),
        fill = 'none'
        ))

# Refine layout:
# layout: bg white, grid gray, no legend (-> use inkscape)
time_fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
time_fig.update_xaxes(
    gridcolor = "lightgray",
    zerolinecolor ="lightgray"

    )
time_fig.update_yaxes(
    tickmode = 'array',
    tickvals = Internal_size[np.round(np.linspace(0, len(Internal_size) - 1, 5)).astype(int)]/Internal_size[-1], # set those to align with the internalDim/maxInternalDim
    ticktext = Internal_size[np.round(np.linspace(0, len(Internal_size) - 1, 5)).astype(int)],
    gridcolor = "lightgray",
    zerolinecolor = "lightgray",
    side = "right"
)
time_fig.show()
print("Values for right axis (internalSize): {}".format(Internal_size[np.round(np.linspace(0, len(Internal_size) - 1, 5)).astype(int)]))
print("Values for left axis (time): {}".format(Internal_size[np.round(np.linspace(0, len(Internal_size) - 1, 5)).astype(int)]/Internal_size[-1]))
# print image as plain svg (-> import into inkscape to draw labels, annotations etc.)
print('start writing image')
time_fig.write_image("../../writing/thesis/diagrams/Time_N{}_U{}.svg".format(N_max,internalDim))
print('image 2 written')