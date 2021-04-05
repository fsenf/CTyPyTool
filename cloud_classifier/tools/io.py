import h5py 
import xarray as xr
import numpy as np



def read_mask(filename, dims, coords):
    """
    Reads mask-data from h5 file and converts it into xr.array       
    
    Parameters
    ----------
    filename : string
        Filename of the mask-data
        
    dims : list
        Name of the dimensions

    coords : dict
        Coordinate values for the mask

    """
    mask_data = h5py.File(filename, 'r')
    mask_xr = xr.Dataset()

    for key in mask_data.keys():
        if key == "_source":
            continue
        m = xr.DataArray([row for row in mask_data[key]], dims = dims, coords = coords, name = key + "_mask")
        mask_xr[key + "_mask"] = m
    return mask_xr



def extract_learning_data(data, labels, indeces, hour = 0): 
    input_data = []
    output_data = []
    for variable in data.variables:
         if "bt" in variable:
            masked_channel = np.array(data[variable])[hour,indeces[0],indeces[1]].flatten() 
            input_data.append(masked_channel)
    input_data = np.array(input_data)
    sp = input_data.shape
    input_data = input_data.flatten().reshape(sp[1],sp[0], order='F')
    
    output_data = np.array(labels["CT"])[hour,indeces[0], indeces[1]].flatten()
    return input_data, output_data


def clean_data(data, labels, indeces):
    counter = 0
    valid = np.ones(len(indeces[0]), dtype=bool)
    for i in range(len(indeces[0])):
        if np.isnan(labels[i]):
            valid[i] = False
            counter += 1
        elif np.isnan(data[i]).any():
            valid[i] = False
            counter += 1
    cleaned_indeces = np.array([indeces[0][valid], indeces[1][valid]])
    cleaned_data = data[valid,:]
    cleaned_labels = labels[valid]
    print("%i positions deleted for including nan-values." % counter)

    return cleaned_data, cleaned_labels, cleaned_indeces
