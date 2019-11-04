"""
Author: Laetitia Papaxanthos
Creation: 10.01.19
"""
from __future__ import division

import os
import numpy as np
import argparse
import pickle
from os.path import join

from utils import get_scores, load_results, early_stopping, get_ensemble_mean,\
	get_ensemble_variance

# The arguments passed to this script are defined below.
parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str)
parser.add_argument('--model_path', type=str)
parser.add_argument('--output_path', type=str)
args = parser.parse_args()

# Set the paths to input and output data.
input_path = args.input_path
model_path = args.model_path
output_path = args.output_path

# Load ground truth target values.
ground_truth_targets_val = np.load(join(input_path, 'targets_validation.npy'))
ground_truth_targets_tst = np.load(join(input_path, 'targets_test.npy'))

# Get model variables.
models_env = os.environ['models_exp']
seed_ensemble_env = os.environ['seed_ensemble_exp']
models = np.array(models_env.split(':')).astype(int)
seed_ensemble = np.array(seed_ensemble_env.split(':')).astype(int)

# Load the results of the ensemble in the validation set. 
predictions_mean_val, _ = load_results(models, 
	seed_ensemble, model_path, data='val')

# Find the best epoch in the validation set.
ensemble_mean_val = get_ensemble_mean(predictions_mean_val)
r2_val = get_scores(ensemble_mean_val, ground_truth_targets_val, data='val')
best_epoch = early_stopping(r2_val) 

# Load the results of the ensemble in the test set and compute mean and variance. 
predictions_mean_tst, predictions_variance_tst = load_results(models, 
	    seed_ensemble, model_path, data='tst', n_epoch=best_epoch)
ensemble_mean_tst = get_ensemble_mean(predictions_mean_tst)
ensemble_variance_tst = get_ensemble_variance(predictions_mean_tst, 
        predictions_variance_tst)
r2_tst, rmse_tst, mae_tst, xfold2_tst = get_scores(ensemble_mean_tst, 
        ground_truth_targets_tst, data='tst')

# Save table containing performance metrics in the test set.
with open(join(output_path, 'prediction_performance.txt'), 'w') as f_out:
    f_out.write('data\tr2\trmse\tmae\t%-within-2-fold\n')
    f_out.write('test set\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n'.format(
	r2_tst, rmse_tst, mae_tst, xfold2_tst))
    f_out.close()

# Save predicted targets and standard deviations of sequences in the test set.
np.savetxt(join(output_path, 'predicted_targets.txt'), 
        ensemble_mean_tst, fmt='%s')
np.savetxt(join(output_path, 'predicted_uncertainty.txt'), 
        np.sqrt(ensemble_variance_tst), fmt='%s')
