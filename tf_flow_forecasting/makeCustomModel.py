#%%
# system imports
from __future__ import print_function
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import sys
import datetime
import time
import multiprocessing
from pathlib import Path
from shutil import copy2

multiprocessing.cpu_count()

# local imports
sys.path.append('../') # add root code folder to search path for includes

from advection.Log import *
from data_generator import *
import advection.parameter as pm


optimizer = 'adam'
loss = tf.keras.losses.mean_squared_error
metrics = ['accuracy','mse','mae']

def conventionalLSTMModel():
    init_std = pm.ini_weight
    initializer = tf.random_normal_initializer(
    mean=0, stddev=init_std, seed=None
    )
    print('stddev={}'.format(np.sqrt(init_std/((pm.steps*pm.basis_M*pm.basis_M+pm.steps*pm.basis_N*pm.basis_N)/2))))
    initializer = tf.keras.initializers.VarianceScaling(
        scale=init_std,mode='fan_avg'
    )

    inputs = tf.keras.Input(shape=(pm.steps, pm.basis_M, pm.basis_M))
    # branch leftcomparably
    x2 = inputs[:,pm.steps-1,:,:]
    # branch right
    internalSize = pm.internalSize 
    x1 = tf.keras.layers.Reshape((pm.steps, pm.basis_M**2))(inputs) # linearizes input
    x1 = tf.keras.layers.Dense(internalSize, activation=tf.keras.activations.tanh, kernel_initializer=initializer)(x1) 
    x1 = tf.keras.layers.LSTM(internalSize, return_sequences=True, kernel_initializer=initializer)(x1)
    x1 = tf.keras.layers.Dense(internalSize, kernel_initializer=initializer)(x1) 
    x1 = tf.keras.layers.LSTM(internalSize, return_sequences=True, kernel_initializer=initializer,recurrent_initializer=initializer)(x1)
    x1 = tf.keras.layers.Dense(internalSize, kernel_initializer=initializer)(x1)
    x1 = tf.keras.layers.LSTM(internalSize, kernel_initializer=initializer, recurrent_initializer=initializer)(x1) # remove time dim
    x1 = tf.keras.layers.Dense(internalSize, activation=tf.keras.activations.tanh, kernel_initializer=initializer)(x1) 
    x1 = tf.keras.layers.Dense(pm.basis_M**2, kernel_initializer=initializer)(x1) # shrink dim to M^2 # no activation alows for arbitary output 
    x1 = tf.keras.layers.Reshape((pm.basis_M, pm.basis_M))(x1)
    # Skip connection ends here
    outputs = tf.keras.layers.Add()([x2,x1])

    simple_lstm_model = tf.keras.Model(inputs=inputs, outputs=outputs, name="LSTM")
    simple_lstm_model.compile(optimizer=optimizer, loss=loss , metrics=metrics)

    return simple_lstm_model

def tinyLSTMModel():
    init_std = pm.ini_weight
    initializer = tf.random_normal_initializer(
    mean=0, stddev=init_std, seed=None
    )
    print('stddev={}'.format(np.sqrt(init_std/((pm.steps*pm.basis_M*pm.basis_M+pm.steps*pm.basis_N*pm.basis_N)/2))))
    initializer = tf.keras.initializers.VarianceScaling(
        scale=init_std,mode='fan_avg'
    )

    internalSize = pm.internalSize 
    inputs = tf.keras.Input(shape=(pm.steps, pm.basis_M, pm.basis_M))
    # branch left
    x2 = inputs[:,pm.steps-1,:,:]
    # branch right
    # Transitioning to internal size [S,M**2]
    x1 = tf.keras.layers.Reshape((pm.steps, pm.basis_M**2))(inputs) 
    x1 = tf.keras.layers.Flatten()(x1)
    x1 = tf.keras.layers.Dense(pm.basis_M*pm.steps, activation=tf.keras.activations.tanh, kernel_initializer=initializer)(x1) 
    x1 = tf.keras.layers.Reshape((pm.steps, pm.basis_M))(x1)
    # Encoding and dropping of time dimension
    x1 = tf.keras.layers.LSTM(internalSize, return_sequences=False, kernel_initializer=initializer)(x1)
    # Transitioning to outputshape [M,M]: 
    x1 = tf.keras.layers.Dense(pm.basis_M**2, kernel_initializer=initializer)(x1)
    x1 = tf.keras.layers.Reshape((pm.basis_M, pm.basis_M))(x1)
    # Skip connection ends here
    outputs = tf.keras.layers.Add()([x2,x1]) 

    tiny_lstm_model = tf.keras.Model(inputs=inputs, outputs=outputs, name="LSTM")
    tiny_lstm_model.compile(optimizer=optimizer, loss=loss , metrics=metrics)
    return tiny_lstm_model

def Conv3DLSTMModel():
    init_std = pm.ini_weight
    initializer = tf.random_normal_initializer(
    mean=0, stddev=init_std, seed=None
    )
    print('stddev={}'.format(np.sqrt(init_std/((pm.steps*pm.basis_M*pm.basis_M+pm.steps*pm.basis_N*pm.basis_N)/2))))
    initializer = tf.keras.initializers.VarianceScaling(
        scale=init_std,mode='fan_avg'
    )

    internalSize = pm.internalSize 
    inputs = tf.keras.Input(shape=(pm.steps, pm.basis_M, pm.basis_M))
    # branch left
    x2 = inputs[:,pm.steps-1,:,:] # theoretically branch left could use dense as well. (probably bad idea)
    # branch right
    # Encoding|Transition layers:
    #x1 = tf.expand_dims(inputs, -1)
    #x1 = tf.keras.layers.Conv3D(1,3,input_shape=(pm.steps, pm.basis_M, pm.basis_M,1),data_format='channels_last',padding='same')(x1)
    x1 = tf.keras.layers.Reshape((pm.steps, pm.basis_M**2))(inputs) 
    # Dropping of time
    x1 = tf.keras.layers.LSTM(internalSize, return_sequences=False, kernel_initializer=initializer)(x1)
    # Transitioning to outputshape [M,M]: 
    #x1 = tf.keras.layers.Dense(pm.basis_M**2,activation='tanh', kernel_initializer=initializer)(x1)
    x1 = tf.keras.layers.Dense(pm.basis_M**2, kernel_initializer=initializer)(x1)
    x1 = tf.keras.layers.Reshape((pm.basis_M, pm.basis_M))(x1)
    # Skip connection ends here
    outputs = tf.keras.layers.Add()([x2,x1]) 
    #outputs = x1
    #TODO: return x2 uncompromised
    #outputs = x2
    tiny_lstm_model = tf.keras.Model(inputs=inputs, outputs=outputs, name="LSTM")

    # available loss and metric:
    tiny_lstm_model.compile(optimizer=optimizer, loss=loss , metrics=metrics)

    return tiny_lstm_model

def Conv3DLSTMModelMultiF():
    init_std = pm.ini_weight
    #initializer = tf.zeros_initializer()
    initializer = tf.random_normal_initializer(
    mean=0, stddev=init_std, seed=None
    )
    #stddev = sqrt(2 / (fan_in + fan_out))#glorot
    #stddev = sqrt(scale / (fan_in+fan_out)/2) VarianceScaling scale = 4 for glorot behaviour?
    print('stddev={}'.format(np.sqrt(init_std/((pm.steps*pm.basis_M*pm.basis_M+pm.steps*pm.basis_N*pm.basis_N)/2))))
    initializer = tf.keras.initializers.VarianceScaling(
        scale=init_std,mode='fan_avg'
    )

    internalSize = pm.internalSize 
    inputs = tf.keras.Input(shape=(pm.steps, pm.basis_M, pm.basis_M))
    # branch left
    x2 = inputs[:,pm.steps-1,:,:] # theoretically branch left could use dense as well. (probably bad idea)
    # branch right
    # Encoding|Transition layers:
    x1 = tf.expand_dims(inputs, -1)
    x1 = tf.keras.layers.Conv3D(3,3,input_shape=(pm.steps, pm.basis_M, pm.basis_M, 1),data_format='channels_last',padding='same')(x1)
    # split filters and input into lstm separately
    #x10 = tf.squeeze(x1[:,:,:,:,0])
    #x11 = tf.squeeze(x1[:,:,:,:,1])
    #x12 = tf.squeeze(x1[:,:,:,:,2])
    x10 = x1[:,:,:,:,0]
    x11 = x1[:,:,:,:,1]
    x12 = x1[:,:,:,:,2]
    x10 = tf.keras.layers.Reshape((pm.steps, pm.basis_M**2))(x10) 
    x11 = tf.keras.layers.Reshape((pm.steps, pm.basis_M**2))(x11) 
    x12 = tf.keras.layers.Reshape((pm.steps, pm.basis_M**2))(x12) 
    # Dropping of time
    x10 = tf.keras.layers.LSTM(internalSize, return_sequences=False, kernel_initializer=initializer)(x10)
    x11 = tf.keras.layers.LSTM(internalSize, return_sequences=False, kernel_initializer=initializer)(x11)
    x12 = tf.keras.layers.LSTM(internalSize, return_sequences=False, kernel_initializer=initializer)(x12)
    #x1 = tf.reshape(tf.stack([x10,x11,x12]), [-1])
    x1 = tf.concat([x10,x11,x12], 1)
    # Transitioning to outputshape [M,M]: 
    x1 = tf.keras.layers.Dense(pm.basis_M**2, kernel_initializer=initializer)(x1)
    x1 = tf.keras.layers.Reshape((pm.basis_M, pm.basis_M))(x1)
    # Skip connection ends here
    outputs = tf.keras.layers.Add()([x2,x1]) 
    #outputs = x1
    #TODO: return x2 uncompromised
    #outputs = x2
    tiny_lstm_model = tf.keras.Model(inputs=inputs, outputs=outputs, name="LSTM")

    # available loss and metric:
    tiny_lstm_model.compile(optimizer=optimizer, loss=loss , metrics=metrics)

    return tiny_lstm_model

showShapes = True
showNames = True
tf.keras.utils.plot_model(Conv3DLSTMModelMultiF(), to_file='../../writing/thesis/diagrams/PGN_MARCH_ConvMultiF_I{}_M{}_N{}_.png'.format(pm.internalSize, pm.basis_M, pm.basis_N), show_shapes=showShapes, show_layer_names=showNames)
tf.keras.utils.plot_model(conventionalLSTMModel(), to_file='../../writing/thesis/diagrams/PGN_MARCH_LSTM_I{}_M{}_N{}_.png'.format(pm.internalSize, pm.basis_M, pm.basis_N),       show_shapes=showShapes, show_layer_names=showNames)
tf.keras.utils.plot_model(tinyLSTMModel(),         to_file='../../writing/thesis/diagrams/PGN_MARCH_TinyLSTM_I{}_M{}_N{}_.png'.format(pm.internalSize, pm.basis_M, pm.basis_N),   show_shapes=showShapes, show_layer_names=showNames)
tf.keras.utils.plot_model(Conv3DLSTMModel(),       to_file='../../writing/thesis/diagrams/PGN_MARCH_ConvLSTM_I{}_M{}_N{}_.png'.format(pm.internalSize, pm.basis_M, pm.basis_N),   show_shapes=showShapes, show_layer_names=showNames)
# Use to plot model architecurues:
#conventionalLSTMModel().summary()
#tinyLSTMModel().summary()
#Conv3DLSTMModel().summary()
#Conv3DLSTMModelMultiF().summary()
print('plotting done')
