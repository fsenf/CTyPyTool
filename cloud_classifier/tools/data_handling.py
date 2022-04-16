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
    mask_data = h5py.File(params["mask_file"], "r")
    m = xr.DataArray(
        [row for row in mask_data[params["mask_key"]]], name=params["mask_key"]
    )
    masked_indices = np.where(m == params["mask_sea_coding"])
    return masked_indices


def create_training_vectors(params, training_sets, masked_indices, verbose):
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

    if not training_sets:
        print("No training data added.")
        return

    # Get vectors from all added training sets
    vectors, labels = td.sample_training_sets(
        training_sets=training_sets,
        n_samples=params["samples"],
        hours=params["hours"],
        indices=masked_indices,
        input_channels=params["input_channels"],
        ct_channel=params["cloudtype_channel"],
        verbose=verbose,
    )

    # Remove nan values
    vectors, labels = td.clean_training_set(vectors, labels)

    if params["difference_vectors"]:
        # create difference vectors
        vectors = td.create_difference_vectors(vectors, params["original_values"])

    if params["nwcsaf_in_version"] == "auto":
        params["nwcsaf_in_version"] = nwc.check_nwcsaf_version(labels, verbose=verbose)

    # Check if
    labels = nwc.switch_nwcsaf_version(
        labels, params["nwcsaf_out_version"], params["nwcsaf_in_version"]
    )
    td.merge_labels(labels, params["merge_list"])
    return vectors, labels


def create_input_vectors(filename, params, indices, hour=0, verbose=True):
    """
    Extracts feature vectors from given NETCDF file at a certain hour.


    Parameters
    ----------
    filename : string
        Filename of the sattelite data

    hour : int
        0-23, hour of the day at which the data set is read

    Returns
    -------
    tuple of numpy arrays
        Array containig the test vectors and another array containing the indices those vectors belong to

    """
    sat_data = xr.open_dataset(filename)
    if indices is None:
        # get all non-nan indices from the first layer specified in input channels
        indices = np.where(~np.isnan(sat_data[params["input_channels"][0]][0]))
        print("No mask indices given, using complete data set")

    vectors = td.extract_feature_vectors(
        sat_data, indices, hour, params["input_channels"]
    )
    vectors, indices = td.clean_test_vectors(vectors, indices)
    if params["difference_vectors"]:
        vectors = td.create_difference_vectors(vectors, params["original_values"])
    if verbose:
        print("Input vectors created!")
    return vectors, indices


def save_training_data(vectors, labels, project_path):
    """
    Saves a set of training vectors and labels

    Parameters
    ----------
    project_path : string
        Path of the project folder.

    vectors : array like
        The feature vectors of the training set.

    labels : array like
        The labels of the training set
    """
    filename = os.path.join(project_path, "data", "training_data")

    dump([vectors, labels], filename)


def load_training_data(project_path):
    """
    Loads a set of training vectors and labels

    Parameters
    ----------
    project_path : string
        Path of the project folder.

    Returns
    -------
    Tuple containing a set of training vectors and corresponing labels
    """
    filename = os.path.join(project_path, "data", "training_data")
    vectors, labels = load(filename)
    return vectors, labels
