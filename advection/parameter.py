import multiprocessing
import numpy as np
################################################################################
# Parameter set up
################################################################################

# Flow data
steps = 8               # number of input steps S
fps = 128               # frames per second during integration
creational_steps = 128  # T = creational_steps/fps, number of training steps
train_steps = creational_steps # makes advanced validation simpler
cache_size = 12         # number of training flows
val_samples = 12        # number of validation flows
valStepOffset = 0       # NOT WORKING USE 0
numOffsets = 2          # NOT WORKING USE 1

# Flow generation
basis_M = 6             # number of low frequncy basis
basis_N = 12            # number of low frequncy basis
epsilonRange = basis_M  # only offset the M basis used (especially usefull when only w_k<M is none-zero initially)
visc = 0                # viscosity
eps = 0.1               # epsilon offset for flows: CAREFUL WITH HIGHER DIMENSIONES scales crazy

# network parameter
internalSize = basis_M**2 # internal size of the network

# training parameter:
BATCH_SIZE = 64         # 64
EPOCHE_STEPS = 64       # 64
ValSteps = 8            # 8
EPOCHS = 2              # Number of epoches per super epoche
SuperEpochs = 2         # Can be set to one, is only used for logging and network snapshots

shuffle_buffer = 100    # shuffel buffer for data shuffel during training
queueSize = BATCH_SIZE*4# can be computed since each batch takes (steps+1)*M^2*8 bytes + C

# optimizer
minDel = 1e-20          # stop after PATIENCE training steps without improvement above minDel!
patience = 60          # set to > Epoches to avoid early stopping

ini_weight = 1e0 # what was this originally? 1 seems rather big

MULTIPROC = True # Flag for training: TODO: is this doing anything usefull?

THREAD_NUM = multiprocessing.cpu_count()-1 # spare one core for myself

# validation parameter (advanced stuff, not needed on a daily basis)
loadModel = False       # this will load the model instead of training it
retrainModel = False    # this will load the model and retrain it (possibly on a different data set or similar)
ipython = False         # either log to file or to ipython console (both is not working very well)

# working name to save model with
workingName = "exampleModel_r" # This is the name of the network currently in training!
# workingName is used for triaining, validation and visualization, keep constant for most of the time

# Validation

# TODO: implement logic to get the last meningful model trained (ie 500 from 100 to 500 but not 1000)
run = 4   # choose snap shot for validation/visualization

# paths (keep as is)
pathToData = './trainingSets/'
ModelPath = pathToData + workingName + str(run)
initialFlowFile = '../advection/initialFlows/M{}_N{}_baseFlow.npy'.format(basis_M,basis_N)
