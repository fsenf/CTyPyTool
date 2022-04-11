import numpy as np
import xarray as xr
import h5py
import re
import os
from joblib import dump, load
import warnings


import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs

import tools.training_data as td
import tools.file_handling as fh
import tools.nwcsaf_tools as nwc
import tools.confusion as conf

#
import importlib
importlib.reload(td)
importlib.reload(fh)
importlib.reload(nwc)


def set_indices_from_mask(params):
    """
    Sets indices according to a selected mask

    Reads mask-data from h5 file and converts it into xr.array. From this data the indices corresponding
    to the selected mask are extracted and saved

    params : dict
        Dictionary of project parameters

    """
    mask_data = h5py.File(params["mask_file"], 'r')
    m = xr.DataArray([row for row in mask_data[params["mask_key"]]],
                     name = params["mask_key"])
    masked_indices = np.where(m == params["mask_sea_coding"])
    return masked_indices


def create_training_vectors(params, training_sets, masked_indices):
    """
    Creates a set of training vectors from NETCDF datasets.

    Samples a set of satellite data and corresponding labels at samples random positions
    for each hour specified.


    Parameters
    ----------
    params : dict
        Dictionaty of project parameters
    training_sets  : list of string tuples
        List of tuples containing the filenames for training data and corresponding labels
    masked_indices : numpy array

    Returns
    -------
    tuple of numpy arrays
        Arrays containig the training vectors and corresponding labels

    """

    if (not training_sets):
        print("No training data added.")
        return

    # Get vectors from all added training sets
    vectors, labels = td.sample_training_sets(training_sets, masked_indices, params["samples"],
                                              params["hours"], params["input_channels"],
                                              params["cloudtype_channel"], verbose = False)

    # Remove nan values
    vectors, labels = td.clean_training_set(vectors, labels)

    if (params["difference_vectors"]):
        # create difference vectors
        vectors = td.create_difference_vectors(vectors, params["original_values"])

    if (params["nwcsaf_in_version"] == 'auto'):
        params["nwcsaf_in_version"] = nwc.check_nwcsaf_version(labels, verbose = False)

    # Check if
    labels = nwc.switch_nwcsaf_version(labels, params["nwcsaf_out_version"], params["nwcsaf_in_version"])
    td.merge_labels(labels, params["merge_list"])
    return vectors, labels
