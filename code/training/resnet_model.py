'''
Author: Laetitia Papaxanthos
Creation: 10.01.19
'''

from __future__ import division

import tensorflow as tf
import numpy as np

from keras import layers, regularizers
from keras import backend as K


eps_ = 10**-6
eps = 10**-9

class UncerEstim():
    
    def __init__(self, args):
        self.args = args
        self.loss_func = self.custom_objective()
    
    def nll_score(self, alpha_beta_groundtruth):
        """ Calculates the average negative log-likelihood over input samples"""
        dst = tf.distributions.Beta(alpha_beta_groundtruth[0], 
                                    alpha_beta_groundtruth[1])
        target = K.clip(alpha_beta_groundtruth[2], eps_, 1 - eps_)
        return K.mean(-dst.log_prob(target))

    def residual_block(self, y, n_in, n_out, kmer_size=[3,3], stride=1):
        """ Defines the residual block, composed of two convolutional layers"""
        shortcut_in = y  
        
        conv1D_1 = layers.Conv1D(
	        n_in, 
	        kernel_size=kmer_size[0], 
	        strides=stride, 
	        kernel_regularizer=regularizers.l2(self.args.weight_decay), 
	        padding='same')
        BN_1 = layers.BatchNormalization()
        y_in = conv1D_1(y)
        y_in = BN_1(y_in)
        y_in = layers.LeakyReLU()(y_in)
        
        conv1D_2 = layers.Conv1D(
		n_out, 
		kernel_size=kmer_size[1],  
		strides=1, 
                kernel_regularizer=regularizers.l2(self.args.weight_decay), 
		padding='same')
        BN_2 = layers.BatchNormalization()
        y_in = conv1D_2(y_in)
        y_in = BN_2(y_in)                   

        if shortcut_in.shape[2] != n_out:
            conv1D_sh = layers.Conv1D(
		n_out, 
		kernel_size=1,  
		strides=stride, 
                kernel_regularizer=regularizers.l2(self.args.weight_decay), 
		padding='same')
            BN_sh = layers.BatchNormalization()
            shortcut_in = conv1D_sh(shortcut_in)
            shortcut_in = BN_sh(shortcut_in)              

        y_in = layers.add([shortcut_in, y_in])
        y_in = layers.LeakyReLU()(y_in)
        return y_in
    
    def core_model(self, x_in):
        """Defines the deep learning model"""
        n_modules = len(self.args.n_blocks_res)
        # Implements the sequence of residual blocks.  
        for module in range(n_modules):
            for block in range(self.args.n_blocks_res[module]):
                stride = 1
                x_in = self.residual_block(
			x_in, 
			int(self.args.n_filters_res[module * 2]), 
			int(self.args.n_filters_res[module * 2 + 1]), 
                        kmer_size=[int(self.args.kmer_sizes_res[module * 2]),
				   int(self.args.kmer_sizes_res[module * 2 + 1])], 
			stride=stride)
        # Flattens the last convolutional layer's output.
        x_in = layers.Flatten()(x_in)
        
	# Implements the two first fully-connected layers.
        dense_alpha = layers.Dense(
		self.args.n_units_mlp, 
		kernel_regularizer=regularizers.l2(self.args.weight_decay))
        BN_alpha = layers.BatchNormalization() 
        alpha_hidden = dense_alpha(x_in)
        alpha_hidden = BN_alpha(alpha_hidden)
        alpha_hidden = layers.LeakyReLU()(alpha_hidden)

        dense_beta = layers.Dense(
		self.args.n_units_mlp, 
		kernel_regularizer=regularizers.l2(self.args.weight_decay))
        BN_beta = layers.BatchNormalization()
        beta_hidden = dense_beta(x_in)
        beta_hidden = BN_beta(beta_hidden)
        beta_hidden = layers.LeakyReLU()(beta_hidden)
        
        # Implements the two fully-connected layers that lead to the two output values.
        dense_output_alpha = layers.Dense(1, activation='softplus')
        dense_output_beta = layers.Dense(1, activation='softplus') 
        alpha = dense_output_alpha(alpha_hidden)
        beta = dense_output_beta(beta_hidden)
        add_epsilon = layers.Lambda(lambda x: x + eps)
        self.alpha = add_epsilon(alpha)
        self.beta = add_epsilon(beta)
            
        
    def sequence_model(self, inputs, targets):
        """Computes output"""
        self.core_model(x_in=inputs)
        self.nll = layers.Lambda(self.nll_score)(
		[self.alpha, self.beta, targets])
        return layers.Concatenate()([self.alpha, self.beta]) 
        
    def custom_objective(self):
        """Computes cost function"""
        nll_loss = self.nll_score
        def loss_function_in(y_true, y_pred):
            value_alpha, value_beta = layers.Lambda(
		lambda x: tf.split(x, 2, axis=1))(y_pred)
            loss = nll_loss(
		[value_alpha, value_beta, y_true])
            return loss
        return loss_function_in
    
