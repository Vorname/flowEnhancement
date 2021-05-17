Minimal documenation for flow enhancement via neural networks in python, tensorflow.

General structure:

advection/
	- files for conventional integration (Code adapted from Mirko Ebert)
	- analysis of flows
	- construction of suitable intial train flow (base flow)
	- parameter for BOTH advection/ and tf_flowforecasting/
	- visualization via LIC (gifs needs to be assembled by hand)

tf_flowforecasting/
	- data generation		# used to generate train and validation flows
	- data generator		# used to read and slice train data during training
	- network architecture		# several network models
	- network training		# training of the model
	- network validation		# validation 
	- validation visualization	# visualizaition of validation

Sine_experiments/ is a stanalone project not thouroughly documented. Use at your own risk.

To train a network guarantee the following directory structure:

advection/
	initialFlows/			# contains base flow at T0
	LICM/				# output for LIC, optional
	LICN/				# output for LIC, optional
tf_flowforecasting/
	flowData/			# contains the flow data used for training and validation
	trainingSets/			# contains snapshots of (partly) trained models and according validation data

Following we assume unchanged prameters in advection/parameter.py.

1. Create base flow at T0 by executing compareBaseFlow.py. This creates M6_N12_baseFlow.npy used for data generation.
(Caveat: working directory must be advection/ NOT tf_flowforecasting)

2. Switch into tf_flowforecasting/ and execute dataGeneration.py
This creates the following files in tf_flowforecasting/flowData/M6_N12_C128_S12_V12_E0.1
	train_0-11.npy 			# flow in N
	train_stepWise_M_0-11.npy	# flow in M, each step integrated from N flwo
	val_0-11.npy			# flow in N
	val_stepWise_M_0-11.npy		# flow in M, each step integrated from N flwo
Train and Validation are created identical and automatically used by training for val_loss and early stopping.

3. Train the data by executing training.py
This creates two snap shots in tf_flowforecasting/ 
	exampleModel_r2/
	exampleModel_r4/
The first trained for 2 epoches the second for 4 epoches (reusing the first one).
We recommend EPOCHES = 100 and SUPEREPOCHES = 10 for a start. This toy example works with EFOCHES=SUPEREPOCHES=2 for the sake of a quick example.
The trained models contain
	iterative/			# for validation data
	stepWise/			# for validation data
	trainedModel/			# the model in tf format (don't touch unless you know tf)
	logfile.log			# dump of the training lof
	makeCustomModel.py		# copy of the file containing the exact architecuture used for creation
	parameter.py			# copy of the parameterisation used for the training
Caveat: the last super epoche might not be trained for the full number of EPOCHES because of early stopping and not all Superepoches are guarenteed to be trained.
Control this behaviour via patience and minDel in parameter file.

3.1 (Optional) Change architecture
	Model architectures are defined in makeCustomModel.py
	To switch models modify line 120 (example: simple_lstm_model = Conv3DLSTMModel())
	(Model architecture is not exposed to parameter file)

4. Validate a single model snap shot by executing validation.py
Validation runs the model on several different modes, most notably the iterative one. 
Validation is computationaly demanding and might take a long time. (often more than training/data generation)
Parameter are not exposed to parameter file but can be set in the main part.
This cerates several validation files in the according trained model directory (here exampleModel_r4/)
Scoring is done in the next step, visualization.

5. Visualization is done by executing visualisation.py
	This creates several plots and does the scoring as well.

CAVEAT validation and visualisation contain a lot of parameters not exposed via parameter.py and are neither as concise nor as well documented as data generation and training are
Currently iterative validation works on the fly only for T0 as start point. This can be manually tweaked to use several start points for iterative validation via parmater.py but is cumbersome and error prone.
It should be fairly straighforward to generalize the concept of several iterative start points and include this in the validation.

Good luck and have fun!
	

