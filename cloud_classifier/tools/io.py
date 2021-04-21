import h5py 
import xarray as xr
import numpy as np


def read_h5mask(filename):
    """
    Reads mask-data from h5 file and converts it into xr.array       
    
    Parameters
    ----------
    filename : string
        Filename of the mask-data
        
    # dims : list
    #     Name of the dimensions

    # coords : dict
    #     Coordinate values for the mask

    """
    mask_data = h5py.File(filename, 'r')
    mask_xr = xr.Dataset()

    for key in mask_data.keys():
        if key == "_source":
            continue
        m = xr.DataArray([row for row in mask_data[key]], name = key )
        mask_xr[key] = m
    return mask_xr

