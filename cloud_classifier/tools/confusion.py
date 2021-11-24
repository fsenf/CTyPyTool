import matplotlib.pyplot as plt
import numpy as np
import itertools
import pandas as pd
import warnings

import tools.file_handling as fh
from matplotlib.colors import LinearSegmentedColormap

from sklearn.metrics.cluster import *

def plot_coocurrence_matrix(label_data, 
                            truth_data,
                            normalize=True,
                            title=None,
                            cmap=LinearSegmentedColormap.from_list('', ['white', 'darkgreen']),
                            missing_indices = None,
                            rate = True,
                            save_file = None,
                            show = True,):

    """
    This function prints and plots the Correlation Matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    cm = confusion_matrix(label_data, truth_data)
    fig = plt.figure(figsize=(12,12))
    fig.patch.set_alpha(1)

    _,_,ct_labels = fh.definde_NWCSAF_variables(missing_indices)
    gen_titel = None
    if normalize:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        gen_titel = "Normalized Correlation Matrix"
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if(np.isnan(cm[i,j])):
                cm[i,j] = 0
    else:
        gen_titel = 'Correlation Matrix'

    if (title is None):
        title = gen_titel
                        

 

    #print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontsize = 20)
    #plt.colorbar()
    tick_marks = np.arange(len(ct_labels))
    plt.ylabel('Ground truth labels',fontsize = 16)
    xl = 'Predicted Labels'
    if(rate):
       xl += ", " + fh.get_match_string(label_data, truth_data)
    plt.xlabel(xl, fontsize = 16)

    plt.yticks(tick_marks, ct_labels, fontsize = 12)
    plt.xticks(tick_marks, ct_labels, rotation = 90, fontsize = 12)

    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if(not np.isnan(cm[i,j]) and not cm[i,j] == 0):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center", verticalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize = 10)


    plt.tight_layout()


    if(save_file is not None):
        plt.savefig(save_file, transparent=False)
    if(show):
        plt.show()
    plt.close()




def confusion_matrix(label_data, ground_truth, missing_indices = None):
    """
    creates coocurrecnce matrix
    

    Parameters
    ----------
    label_data, ground_truth: numpy array
        the two label arrays of the same scope from which the cooccurence matrix is created

    """

    d1, d2 = fh.clean_eval_data(label_data, ground_truth)
    _, ct_indices,_ = fh.definde_NWCSAF_variables(missing_indices)
    ct_indices = [int(i) for i in ct_indices]

    cm = np.zeros([len(ct_indices), len(ct_indices)],dtype = int)
    for  i, value in enumerate(d2):
        for j, label in enumerate(ct_indices):
            if d1[i] == label:
                cm[ct_indices.index(value)][j] += 1
    return cm



