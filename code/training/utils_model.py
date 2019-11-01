"""
Author: Laetitia Papaxanthos
Creation: 10.01.19
"""

from __future__ import division

import os
import numpy as np
from os.path import join


def save_model(path, filename, model):
    """Save the weights of the model"""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    model.save(join(path, filename + '.h5')) 


def beta2mean_var(output_beta):
    """Calculate the mean and variance of the beta distribution from the 
    shape parameters"""
    mean = output_beta[:, 0] / (output_beta[:, 1] + output_beta[:, 0])
    variance = (output_beta[:, 0] * output_beta[:, 1]) / (
			(output_beta[:, 0] + output_beta[:, 1])**2
                        * (output_beta[:, 0] + output_beta[:, 1] + 1))
    return mean, variance

# Hyperparameters that were selected on the validation set.
hyperparameters_model_1 = {
         'n_blocks_res': [1, 1, 1],
         'n_filters_res': [64, 64, 64, 64, 64, 64],
         'kmer_sizes_res': [9, 1, 9, 1, 9, 1],
         'n_units_mlp': 64,
         'learning_rate': 0.01,
         'batch_size': 512,
         'weight_decay': 10**(-6)
         }    

hyperparameters_model_2 = {
         'n_blocks_res': [1, 1, 1],
         'n_filters_res': [512, 512, 512, 512, 512, 512],
         'kmer_sizes_res': [10, 1, 10, 1, 10, 1],
         'n_units_mlp': 64,
         'learning_rate': 0.001,
         'batch_size': 512,
         'weight_decay': 10**(-6)
         }    

