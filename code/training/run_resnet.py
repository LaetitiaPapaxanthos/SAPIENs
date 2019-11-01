"""
Author: Laetitia Papaxanthos
Creation: 10.01.19
"""

from __future__ import division

import argparse
import pickle
import os
import json
import sys
from os.path import join
import time

import random
import numpy as np
import tensorflow as tf
from keras import activations, layers, models, optimizers, regularizers
from keras import backend as K

from resnet_model import UncerEstim
from utils_model import beta2mean_var, save_model, \
hyperparameters_model_1, hyperparameters_model_2


# The arguments passed to this script are defined below.
parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str)
parser.add_argument('--output_path', type=str)

parser.add_argument('--seed_ensemble', type=int)

parser.add_argument('--hyperparameters', type=int, default=1)

parser.add_argument('--n_evals', type=int, default=60)
parser.add_argument('--n_epochs_between_eval', type=int, default=5)

parser.add_argument('--save_model', type=bool, default=False)
parser.add_argument('--cuda', type=int, default=3)

args = parser.parse_args()

# Get the hyperparameters from dictionnaries.
hyperparameters = [hyperparameters_model_1 if args.hyperparameters==1 
        else hyperparameters_model_2][0]
args.n_blocks_res = hyperparameters['n_blocks_res']
args.n_filters_res = hyperparameters['n_filters_res']
args.kmer_sizes_res = hyperparameters['kmer_sizes_res']
args.n_units_mlp = hyperparameters['n_units_mlp']
args.learning_rate = hyperparameters['learning_rate']
args.batch_size = hyperparameters['batch_size']
args.weight_decay = hyperparameters['weight_decay']

# Create variables for the paths and create path if necessary.
path_data = args.input_path
output_path = args.output_path
if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)
 
# Start recording time.
tic=time.time()

# Set GPU. 
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)

# Set the seeds.
random.seed(19847)
np.random.seed(args.seed_ensemble)
tf.set_random_seed(args.seed_ensemble + 10)

# Load the sequence files and the target files for the training, validation and test sets.
sequences_tr = np.load(join(path_data, 'sequences_train_onehot.npy'))
sequences_val = np.load(join(path_data, 'sequences_validation_onehot.npy'))
sequences_tst = np.load(join(path_data, 'sequences_test_onehot.npy'))
targets_tr = np.load(join(path_data, 'targets_train.npy'))

# Initialise the lists that will save the output of the model.
shape_parameters_val_to_save = []
shape_parameters_tst_to_save = []
mean_and_variance_val_to_save = []
mean_and_variance_tst_to_save = []

# Build the model. 
US = UncerEstim(args)  
network_input = layers.Input(shape=(sequences_tr.shape[1], 
			            sequences_tr.shape[2]))
network_target = layers.Input(shape=(1,))  
network_output = US.sequence_model(inputs=network_input, 
				   targets=network_target)
model = models.Model(inputs=[network_input], 
                     outputs=[network_output])
loss_func = US.custom_objective()
model.compile(optimizer=optimizers.Adam(lr=args.learning_rate), loss=loss_func) 


# Start training.
for i in range(args.n_evals):
    model.fit(sequences_tr, 
	      targets_tr, 
              batch_size=args.batch_size,
              initial_epoch=i * args.n_epochs_between_eval, 
              epochs=(i + 1) * args.n_epochs_between_eval) 
    
    # Get the predicted shape parameters for the validation and test sets.
    shape_parameters_val = model.predict(sequences_val).squeeze()    
    shape_parameters_tst = model.predict(sequences_tst).squeeze()  
    
    # Tranform the predicted shape parameters into predicted mean and variance.
    mean_val, variance_val = beta2mean_var(shape_parameters_val)
    mean_tst, variance_tst = beta2mean_var(shape_parameters_tst)
    
    # Record the predicted shape parameters, means and variances.
    shape_parameters_val_to_save.append(shape_parameters_val)
    shape_parameters_tst_to_save.append(shape_parameters_tst)
    mean_and_variance_val_to_save.append([mean_val, variance_val])
    mean_and_variance_tst_to_save.append([mean_tst, variance_tst])
    
    if args.save_model == True:
        save_model(join(output_path,'saved_models'), 
                   'model{}_ens{}_eval{}'.format(args.hyperparameters,
		                      args.seed_ensemble, i), 
                    model)

# Records predictions for the training set.
shape_parameters_tr = model.predict(sequences_tr).squeeze() 
mean_tr, variance_tr = beta2mean_var(shape_parameters_tr) 

# End recording time.
toc=time.time()

# Create output directory if it doesnot exit
if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)

# Save results.   
with open(join(output_path, 
		'commandline_args_model{}_ens{}.txt'.format(args.hyperparameters,
		args.seed_ensemble)), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

with open(join(output_path, 
		'target_predictions_model{}_ens{}.pkl'.format(
		args.hyperparameters, args.seed_ensemble)), 'wb') as fb:
    pickle.dump({'shape_parameters_train': [shape_parameters_tr],
                 'mean_and_variance_train': [mean_tr, variance_tr],
                 'shape_parameters_val': shape_parameters_val_to_save,
                 'mean_and_variance_val': mean_and_variance_val_to_save,   
                 'shape_parameters_tst': shape_parameters_tst_to_save,
                 'mean_and_variance_tst': mean_and_variance_tst_to_save,
                 }, fb)
    
