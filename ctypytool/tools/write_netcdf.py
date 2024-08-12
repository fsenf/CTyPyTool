import xarray as xr
import numpy as np
import pathlib
import os

from . import file_handling as fh


def get_reference_filepath(project_path):
    return os.path.join(project_path, "data", "label_reference.nc")


def create_reference_file(project_path, training_sets, cloudtype_channel):
    ref_path = get_reference_filepath(project_path)
    fh.create_subfolders(path=ref_path, contains_filename=True)
    if pathlib.Path(ref_path).is_file():  # check if already exists
        return

    try:
        reference = training_sets[0][1]
    except Exception:
        raise RuntimeError("Can not create reference file. No training data added")

    data = xr.open_dataset(reference)
    for key in data.keys():
        if not key == cloudtype_channel:
            data = data.drop(key)
    data.to_netcdf(path=ref_path, mode="w")


def make_xrData(
    labels, indices, project_path, ct_channel, NETCDF_out=None, prob_data=None
):
    """
    Transforms a set of predicted labels into xarray-dataset

    Parameters
    ----------
    labels : array-like
        Int array of label data

    indices : tuple of array-like
        Indices of the given labels in respect to the coordiantes
        from a reference file

    NETCDF_out : string
        (Optional) If specified, the labels will be written to a
        NETCDF file with this name

    Returns
    -------
    xarray dataset
        labels in the form of an xarray dataset

    """
    reference_file = get_reference_filepath(project_path)
    if not pathlib.Path(reference_file).is_file():
        raise ValueError("Reference file must be created!")

    out = xr.open_dataset(reference_file)

    shape = out[ct_channel][0].shape  # 0 being the hour
    new_data = np.empty(shape)
    new_data[:] = np.nan
    new_data[indices[0], indices[1]] = labels
    out[ct_channel][0] = new_data

    if prob_data is not None:
        shape += (len(prob_data[0]),)
        new_data = np.empty(shape)
        new_data[:] = np.nan
        new_data[indices[0], indices[1]] = prob_data

        new_dims = out[ct_channel][0].dims
        new_dims += ("labels",)
        out["label_probability"] = (new_dims, new_data)

    if NETCDF_out is not None:
        out.to_netcdf(path=NETCDF_out, mode="w")

    return out
