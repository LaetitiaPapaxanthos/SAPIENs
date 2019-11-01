"""
Author: Laetitia Papaxanthos
Creation: 10.01.19
"""

import numpy as np
import matplotlib.pyplot as plt


size_bin = 0.05
percent_outlier = 0.5

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value

def violinplot(ground_truth_targets, predicted_targets):

    # Bin the samples according to their ground truth target values.
    target_binned = []
    idx_outliers_per_bin = []
    target_binned_outliers = []
    for bins in np.arange(0, 1, size_bin):
        if bins == 1 - size_bin:
            idx = np.logical_and(ground_truth_targets >= bins, 
                                 ground_truth_targets <= bins + size_bin)
        else:
            idx = np.logical_and(ground_truth_targets >= bins, 
                                 ground_truth_targets < bins + size_bin)
        low_perc = np.percentile(predicted_targets[idx], percent_outlier)
        up_perc = np.percentile(predicted_targets[idx], 100 - percent_outlier)
        idx_outliers_per_bin.append(np.where(idx)[0][np.logical_or(
                                            predicted_targets[idx] < low_perc, 
                                            predicted_targets[idx] > up_perc)])
        idx_nonoutliers_per_bin = np.logical_and(predicted_targets[idx] >= low_perc, 
                                                 predicted_targets[idx] <= up_perc)
        idx_to_keep_per_bin = np.where(idx)[0][idx_nonoutliers_per_bin]

        target_binned.append(predicted_targets[idx_to_keep_per_bin])
    target_binned_outliers = [predicted_targets[idx] for idx in idx_outliers_per_bin]
    
    # Get summary statistics of the predicted values, per bin.
    percentile_info = np.array([
        np.percentile(target_binned[i], [25, 50, 75]) 
        for i in range(len(target_binned))])
    whiskers = np.array([
        adjacent_values(np.sort(bin_array), q1, q3)
        for bin_array, q1, q3 
        in zip(target_binned, percentile_info[:, 0], percentile_info[:, 2])])
    whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]
    
    # Create the violin plot figure.
    fig = plt.figure(figsize=(18, 11.1255))
    # Plot violins.
    parts = plt.violinplot(target_binned, 
                           showmeans=False, 
                           showmedians=False, 
                           showextrema=False, 
                           widths=size_bin * 0.8, 
                           positions=np.arange(size_bin / 2, 1, size_bin), 
                           points=300,  
                           bw_method=None)
    # Set parameters of the plot.
    for pc in parts['bodies']:
        pc.set_facecolor('blue')
        pc.set_alpha(1)
    # Plot the outliers (scatter plot).
    for it, bin_id in enumerate(np.arange(0, 1, size_bin)):
        plt.scatter([bin_id + 0.025]*len(target_binned_outliers[it]), 
                    target_binned_outliers[it], color='blue', s=0.5)
    # Plot the summary statisics of predicted values, per bin.
    inds = np.arange(0, 1, size_bin) + 0.025
    plt.scatter(inds, percentile_info[:, 1], marker='o', 
                color='white', s=30, zorder=3)
    plt.vlines(inds, percentile_info[:, 0], percentile_info[:, 2], 
               color='k', linestyle='-', lw=5)
    plt.vlines(inds, whiskersMin, whiskersMax, 
               color='k', linestyle='-', lw=1)
    plt.plot([0, 1],[0, 1],'k')

    plt.xlabel('Ground truth $IFP_{0-480min}$ (bins of size 0.05)', fontsize=25)
    plt.ylabel('Predicted  $IFP_{0-480min}$', fontsize=25)

    plt.xticks(np.arange(size_bin / 2, 1, size_bin), 
               ['{:.2f}\n-\n{:.2f}'.format(i * size_bin, (i + 1) * size_bin) 
                for i in range(len(target_binned))], fontsize=12)
    plt.yticks(fontsize=16)
    plt.xlim(0, 1)
    plt.ylim(-0.01, 1.01)

    plt.savefig('Violin_plot.pdf')
    plt.show()



