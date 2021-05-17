# this file is solely for training the network specified in makeCustomModel.py
# trained model and parameters are stored in ./trainingSets
# ./trainingSets contains the trained model (.h5) the model parametrization and the training log

# parameters are set using the parameter.py and makeCustomModel.py 
# prerequisits: trainingData and fitting parametrisation
#%%
# system imports TODO: sort this shit out
from __future__ import print_function
import tensorflow as tf
import sys
import numpy as np
from pathlib import Path
import os as os

from shutil import copy2
import datetime

# local imports
sys.path.append('../') # add root code folder to search path for includes

from data_generator import *
from makeCustomModel import *
# Import parameters
import advection.parameter as pm
print('parmeters loaded')

#%%

# Initialization:

if __name__ == '__main__':
    tf.keras.backend.clear_session()


    overall_time = time.time()
    seenEpoches = 0

    startRun = 0
    if pm.retrainModel == True: # stupid 'fix'
        startRun = 1

    # start the train loop    
    for run in range(startRun, pm.SuperEpochs):
        if run == 0:
            retrain = pm.retrainModel 
        else:
            retrain = True

        timer_start = time.time()

        # create logging and save dir: 
        runName = pm.workingName
        dirPath = "./trainingSets/{}{}".format(runName, (run+1)*pm.EPOCHS)
        dataPath = "./flowData/M{}_N{}_C{}_S{}_V{}_E{}".format(pm.basis_M, pm.basis_N, pm.creational_steps, pm.cache_size, pm.val_samples, pm.eps)


        if not os.path.exists(dataPath):
            print("data path: {} invalid, abort..".format(dataPath))
            exit()
        else:
            print("data path valid")

        Path(dirPath).mkdir(parents=True, exist_ok=True)
        Path(dirPath+"/stepWise").mkdir(parents=True, exist_ok=True)
        Path(dirPath+"/iterative").mkdir(parents=True, exist_ok=True)
        #Path(workingPath).mkdir(parents=True, exist_ok=True)


        # save parameters to savedir''
        copy2("../advection/parameter.py", dirPath)
        print('parameter saved external')


        # Set up logging of console output into file

        class Logger(object):
            def __init__(self):
                self.terminal = sys.stdout
                self.log = open(dirPath+"/logfile.log", "a")

            def write(self, message):
                self.terminal.write(message)
                self.log.write(message)  

            def flush(self):
                #this flush method is needed for python 3 compatibility.
                #this handles the flush command by doing nothing.
                #you might want to specify some extra behavior here.
                pass    

        if not pm.ipython:
            
            sys.stdout = Logger()

        ################################################################################
        # Computation
        ################################################################################

        # Set up generator 
        print("initalizing data")

        generator_function = data_generator(pm.steps, pm.fps, pm.basis_M, pm.basis_N, dirPath = dataPath, viscosity=pm.visc, epsilon=pm.eps, creational_steps=pm.creational_steps, cache_size=pm.cache_size, train_steps=pm.train_steps)

        ds_series = tf.data.Dataset.from_generator(
            generator_function,
            output_types=(tf.float64, tf.float64), 
            output_shapes=((pm.steps, pm.basis_M, pm.basis_M,), (pm.basis_M, pm.basis_M))
            )

        ds_series_train = ds_series.shuffle(pm.shuffle_buffer, reshuffle_each_iteration=False).repeat().padded_batch(pm.BATCH_SIZE,([pm.steps, pm.basis_M, pm.basis_M],[pm.basis_M, pm.basis_M]))

        ds_series_val = ds_series.shuffle(int(pm.shuffle_buffer/10), reshuffle_each_iteration=True).repeat().padded_batch(pm.BATCH_SIZE,([pm.steps, pm.basis_M, pm.basis_M],[pm.basis_M,pm.basis_M]))
        print(ds_series_train,"\n")

        # set up callbacks, logging and model
        
        if not retrain:
            print('creating model')
            simple_lstm_model = Conv3DLSTMModelMultiF()
            print("\n model compiled")
            copy2("./makeCustomModel.py", dirPath)
        else:
            modelPath = pm.pathToData + runName + "{}".format(pm.EPOCHS*run)
            simple_lstm_model = tf.keras.models.load_model(modelPath+'/trainedModel')
        # Plot the model to console
        simple_lstm_model.summary()

        # get model weight norm
        preWeights = simple_lstm_model.get_weights()
        model_norm_preTraining = 0
        for i in range(len(preWeights)):
            model_norm_preTraining = model_norm_preTraining + np.sum(np.abs(preWeights[i]))
        print("Pretraining Model Weights: {}\n".format(model_norm_preTraining))
        # Inital model weigth !=0 because of:
        # Boolean (default True). If True, **add 1 to the bias** of the forget gate at initialization.
        #  Setting it to true will also force bias_initializer="zeros".
        #  This is recommended in Jozefowicz et al..
        # From tf doku

        # train model
        callbacks = [
        keras.callbacks.EarlyStopping(
            # Stop training when `val_loss` is no longer improving
            monitor='val_loss',
            # "no longer improving" being defined as "no better than some value or less"
            min_delta=pm.minDel,
            # "no longer improving" being further defined as "for at least some number of epochs"
            patience=pm.patience,
            verbose=1)]
            
        savePath = dirPath+"/trainedModel"
        
        print("\n training start: \n") 
        history = simple_lstm_model.fit(ds_series_train, validation_data=ds_series_val, validation_steps=pm.ValSteps, steps_per_epoch = pm.EPOCHE_STEPS, callbacks=callbacks, epochs=pm.EPOCHS, use_multiprocessing = pm.MULTIPROC, workers = pm.THREAD_NUM, max_queue_size=pm.queueSize, verbose=2)
        print('save model in {}\n'.format(savePath))

        # create model snap shot
        simple_lstm_model.save(savePath)

        print('model succesfully saved')

        # get model weight norm
        postWeights = simple_lstm_model.get_weights()
        model_norm_postTraining = 0
        model_norm_diff = 0
        for i in range(len(postWeights)):
            model_norm_postTraining = model_norm_postTraining + np.sum(np.abs(postWeights[i]))
            model_norm_diff = model_norm_diff + np.sum(np.abs(preWeights[i]-postWeights[i]))
        print("Posttraining Model Weights: {}\n".format(model_norm_postTraining))
        print("Model weight difference: {}\n".format(model_norm_diff))

        
        timer_end = time.time()
        print("Training took {}sec\n".format(timer_end-timer_start))
        # if early stopping was initiated don't run the next loop, exit here!
        #fd.write("units, prev, loss, error, score, time, epochs\n")
        if not len(history.history['loss']) == pm.EPOCHS:
            print("Earlystopping detected, training halted preemtively")
            break
        # clear the session to free space and fix layer numbering
        tf.keras.backend.clear_session()

    print("Overall Time: {}".format(time.time()-overall_time))
        

# %%
