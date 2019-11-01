'''
Author: Laetitia Papaxanthos
Creation: 10.01.19
'''

from __future__ import division

import numpy as np
import pickle
from os.path import join

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def seq2onehot(sequences, vector_form=True):
    """Transform 2D char array into 2D binary array (n_samples x 4*n_bases) 
    if vector_form is True, or into a 3D binary array 
    (n_samples x n_bases x 4) if vector_form is False"""
    sequences = np.array([list(seq) for seq in sequences])  # list of list of single char
    sequences_onehot = np.stack([(sequences == c).astype(float) for c in 
                        ['A', 'T', 'C', 'G']], axis=2)
    if vector_form:
        sequences_onehot = sequences_onehot.reshape((len(sequences_onehot), -1))
    return sequences_onehot

# Calculale the proportion of predicted targets that are in the X fold
# interval around the ground truth target.
xfold_score = lambda y_true, y_pred, xfold: np.mean(
	np.logical_and(y_pred >= y_true / xfold, y_pred <= xfold * y_true))


def get_ensemble_mean(predictions):
    ensemble_predicted_target = np.mean(predictions, axis=0) 
    return ensemble_predicted_target


def get_ensemble_variance(predictions, variances):
    average = np.mean(predictions, axis=0)
    variance = np.mean(np.square(predictions) + variances, axis=0) \
                - np.square(average)
    return variance


def get_scores(ensemble_predicted_target, true_target, data='val'):
    if data == 'val':
        r2 = np.array([r2_score(true_target, n_pred_target) 
            for n_pred_target in ensemble_predicted_target])
        return r2
    else:
        r2 = r2_score(true_target, ensemble_predicted_target) 
        rmse = np.sqrt(mean_squared_error(true_target, 
                                            ensemble_predicted_target))
        mae = mean_absolute_error(true_target, ensemble_predicted_target)
        xfold2 = xfold_score(true_target, ensemble_predicted_target, xfold=2)
    return r2, rmse, mae, xfold2


def early_stopping(r2):
    index = np.argmax(r2)
    return index


def load_results(models, seed_ensemble, model_path, data='val', n_epoch=None):  
    if data == 'val':
        prediction_results = []
        for m in models:
            for se in seed_ensemble:
                all_results = pickle.load(open(join(model_path, 
			            'target_predictions_model{}_ens{}.pkl'.format(m, se)), 'rb'))
                prediction_results.append(all_results['mean_and_variance_val'])
        prediction_results = np.array(prediction_results)
        predictions_mean = prediction_results[:, :, 0, :]
        predictions_variance = prediction_results[:, :, 1, :]
    elif data == 'tst':
        prediction_results = []
        for m in models:
            for se in seed_ensemble:
                all_results = pickle.load(open(join(model_path, 
			            'target_predictions_model{}_ens{}.pkl'.format(m, se)), 'rb'))
                prediction_results.append(all_results['mean_and_variance_tst'][n_epoch])
        prediction_results = np.array(prediction_results)
        predictions_mean = prediction_results[:, 0, :]
        predictions_variance = prediction_results[:, 1, :]
    return predictions_mean, predictions_variance

