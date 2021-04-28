import numpy as np
import xarray as xr
import random
import h5py 



def read_h5mask(filename):
    """
    Reads mask-data from h5 file and converts it into xr.array
    """
    mask_data = h5py.File(filename, 'r')
    mask_xr = xr.Dataset()

    for key in mask_data.keys():
        if key == "_source":
            continue
        m = xr.DataArray([row for row in mask_data[key]], name = key )
        mask_xr[key] = m
    return mask_xr




def get_mask_indices(filename, mask_name):
    """
    Reads mask from h5-file and returns specified indices
    """
    mask = read_h5mask(filename)
    return np.where(mask[mask_name])


def sample_training_set(filename_data, filename_labels, n = 100, hours = range(24), 
                        indices = None, cDV = True, kOV = True):
    """
    Creates a sample of training vectors from NETCDF datasets


    Samples a set of satellite data and corresponding labels at n random positions 
    for each hour specified. 
    """
    sat_data = xr.open_dataset(filename_data)
    label_data = xr.open_dataset(filename_labels)

    if (indices is None):
        # get all non-nan indices from the 'bt120' layer of the sattelite-data 
        # from first training set at the first hour
        indices = np.where(~np.isnan(sat_data['bt120'][0]))
        print("No mask indices given, using complete range of data")
    
    vectors = None
    labels = np.array([])
    for h in hours:
        # get n random positiobs
        selection = get_sample_positions(indices, n)
        v = extract_trainig_vectors(sat_data, selection, h)
        if(vectors is None):
            # initalize with first batch of vectors
            vectors = v
        else:
            vectors = np.append(vectors, v, axis = 0)
        labels = np.append(labels, exctract_labels(label_data, selection, h), axis = 0)

    # remove nans
    vectors, labels = clean_training_vectors(vectors, labels)
    # calc difference vectors
    if (cDV):
        vectors = create_difference_vectors(vectors, kOV)

    return vectors, labels


def create_test_vectors(filename_data, hour, indices, cDV=True, kOV=True):
    """
    Extracts vectors from NETCDF file for a given set of indices.

    Vectors are created and cleaned of nan-values. If specified difference vectors are created.
    Indidces of the cleaned vector are returned with the vectors

    """
    sat_data = xr.open_dataset(filename_data)

    if (indices is None):
        # get all non-nan indices from the 'bt120' layer of the sattelite-data 
        # from first training set at the first hour
        indices = np.where(~np.isnan(sat_data['bt120'][0]))
        print("No mask indices given, using complete data set")

    vectors = extract_trainig_vectors(sat_data, indices, hour)
    vectors, indices = clean_test_vectors(vectors, indices)
    if (cDV):
        vectors = create_difference_vectors(vectors, kOV)
    return vectors, indices


def exctract_labels(label_data, indices, hour = 0):
    """
    Extract labels from xarray at given indices and time

    Assumes labels are stored under key: "CT"    
    """
    return np.array(label_data["CT"])[hour,indices[0], indices[1]].flatten()

def exctract_labels_fromFile(filename, indices, hour = 0):
    """
    Extract labels from netCDF file at given indices and time
    """
    return exctract_labels(xr.open_dataset(filename), indices, hour)

def get_sample_positions(indices, n):
    """
    Get n random samples from a set of indices    
    """
    selection = np.array(random.sample(list(zip(indices[0],indices[1])),n))
    # adjust selection shape (n,2) --> (2,n)
    s = selection.shape
    selection = selection.flatten().reshape(s[1],s[0],order = 'F')
    return selection


def extract_trainig_vectors(data, indices, hour = 0): 
    """
    Extract training vectors from xr-dataset at given indices and time

    Assumes relevant variables are named as "btX" with X specifing cloud height
    
    """
    vectors = []
    for variable in data.variables:
         if "bt" in variable:
            # remove layer bt039 because of unreliable data
            if "bt039" in variable:
                continue
            masked_channel = np.array(data[variable])[hour,indices[0],indices[1]].flatten() 
            vectors.append(masked_channel)
    vectors = np.array(vectors)
    sp = vectors.shape
    vectors = vectors.flatten().reshape(sp[1],sp[0], order='F')
    return vectors




def clean_training_vectors(vectors, labels):
    """
    Remove vectors and corresponding labels containing nans
    
    """
    valid = ~np.isnan(vectors).any(axis=1)
    valid_l = ~np.isnan(labels)
    valid = np.logical_and(valid, valid_l)
    d = valid.size - vectors[valid].shape[0]
    if (d>0):
          print("Removed " + str(d) + " vectors for containig 'Nan' values")
 
    return vectors[valid], labels[valid]

def clean_test_vectors(vectors, indices):
    """
    Remove vectors containing nans
    Remove corresponding indices
    
    """
    valid = ~np.isnan(vectors).any(axis=1)
    d = valid.size - vectors[valid].shape[0]    
    if (d>0):
        print("Removed " + str(d) + " vectors for containig 'Nan' values")
    return vectors[valid], np.array([indices[0][valid], indices[1][valid]])



def create_difference_vectors(data, keep_original_values = False):
    """
    Transfroms a set of training vectors into a set of inter-value-difference-vectors.    
    """
    new_data = []
    for vector in data:
        new_vector = list(vector) if(keep_original_values) else []
        for i in range(len(vector)):
            for j in range(i+1, len(vector)):
                new_vector.append(vector[i]-vector[j])
        new_data.append(new_vector)
    return np.array(new_data)


def imbed_data(labels, indices, filename):
    """
    Transforms a set of predicted labels into xarray-dataset    
    """
    sat_data = xr.open_dataset(filename)
    # get meta data
    coords = {'lat': sat_data.coords['lat'], 'lon':sat_data.coords['lon']}
    lons = sat_data['lon']
    lats = sat_data['lat']
    dims = ['rows', 'cols']

    new_data = np.empty(lons.shape)
    new_data[:] = np.nan
    new_data[indices[0],indices[1]] = labels
    new_data = xr.DataArray(new_data, dims = dims, coords = coords, name = "CT")
    new_data = new_data.to_dataset()

    return new_data


def write_NETCDF(data, filename):
    data.to_netcdf(path=filename, mode='w')


























#########################################################################################
# def cleanData(data, indices):
#     counter = 0
#     valid = np.ones(len(indices[0]), dtype=bool)
#     for i in range(len(indices[0])):
#         if np.isnan(data[i]).any():
#             valid[i] = False
#             counter += 1
#     cleaned_indices = np.array([indices[0][valid], indices[1][valid]])

#     return cleaned_indices

# def clean_indices(data, indices):
#     """
#     Remove indices pointing to NaN-values in data
    
#     """
#     ci = indices
#     data = data[:][:][ci[0],ci[1]]
#     for v in data.variables:
#         if "bt" in v:
#             for i in range(24):
#                 di = np.array(data[v])[i,ci[0],ci[1]]
#                 #data = data[v][i,ci[0],ci[1]]

#                 ni = np.where(~np.isnan(di))
#                 ci = [ci[0][ni[0]],ci[1][ni[0]]]

#     return ci