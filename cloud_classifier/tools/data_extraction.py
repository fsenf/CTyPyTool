import xarray as xr
import random
import numpy as np


"""
Adds helper-methods for extracting machine learining data from NETCDF datasets
"""


def sample_training_sets(training_sets, n, hours, indices, input_channels,  ct_channel = "CT", verbose = False):
    """
    Creates a sample of training vectors from NETCDF datasets


    Samples a set of satellite data and corresponding labels at n random positions 
    for each hour specified. 
    """

    training_vectors = None
    training_labels = np.array([])

    # sample all added sets
    i = 0
    for t_set in training_sets:
        if(verbose):
            i+=1
            print("Sampling dataset " + str(i) + "/" + str(len(training_sets)) )
        #read data
        sat_data = xr.open_dataset(t_set[0])
        # check if indices have benn selected
        if (indices is None):
            # if not: get all non-nan indices from the first layer specified in input channels
            indices = np.where(~np.isnan(sat_data[input_channels[0]][0]))
            print("No mask indices given, using complete range of data")

        # extract traininig vectors for all hours
        vectors = None
        labels = np.array([])
        # sample all hours
        for h in hours:
            # get n random positions
            selection = get_samples(indices, n)
            # get feature vectors for selection
            v = extract_feature_vectors(sat_data, selection, h, input_channels)
            if(vectors is None):
                # initalize with first batch of vectors
                vectors = v
            else:
                vectors = np.append(vectors, v, axis = 0)
            labels = np.append(labels, 
                extract_labels(t_set[1], selection, h, ct_channel = ct_channel), 
                axis = 0)


        if (training_vectors is None):
            # initalize with first set of vectors
            training_vectors = vectors
        else:
            # append next set of vectors
            training_vectors = np.append(training_vectors, vectors, axis = 0)
        # append labels
        training_labels = np.append(training_labels, labels, axis = 0)

    return training_vectors, training_labels



def get_samples(indices, n):
    """
    Get n random samples from a set of indices    
    """
    selection = np.array(random.sample(list(zip(indices[0],indices[1])),n))
    # adjust selection shape (n,2) --> (2,n)
    s = selection.shape
    selection = selection.flatten().reshape(s[1],s[0],order = 'F')
    return selection



def extract_feature_vectors(data, indices, hour, input_channels): 
    """
    Extract training vectors from xr-dataset at given indices and time
    
    """
    vectors = []
    for channel in input_channels:
        values = np.array(data[channel])[hour,indices[0],indices[1]].flatten() 
        vectors.append(values)

    # reshape from "channels X n" to "n X channels"
    vectors = np.array(vectors)
    sp = vectors.shape
    vectors = vectors.flatten().reshape(sp[1],sp[0], order='F')
    return vectors


def clean_training_set(vectors, labels):
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



def extract_labels(filename, indices, hour = 0, ct_channel = "CT"):
    """
    Extract labels from xarray at given indices and time

    Assumes labels are stored under key: "CT"    
    """
    label_data = xr.open_dataset(filename)
    if (indices is None):
         indices = np.where(~np.isnan(label_data[ct_channel][hour]))
    return np.array(label_data[ct_channel])[hour,indices[0], indices[1]].flatten()


def clean_test_vectors(vectors, indices):
    """
    Remove vectors containing nans while tracking valid indices
     
    """
    valid = ~np.isnan(vectors).any(axis=1)
    d = valid.size - vectors[valid].shape[0]    
    if (d>0):
        print("Removed " + str(d) + " vectors for containig 'Nan' values")
    return vectors[valid], np.array([indices[0][valid], indices[1][valid]])


def make_xarray(labels, indices, reference_filename, labelkey = "CT"):
    """
    returns coordination data from NETCDF file
    """
    data = xr.open_dataset(reference_filename)
    coords = {'lat': data.coords['lat'], 'lon':data.coords['lon']}
    dims = ['rows', 'cols']
    shape = coords['lon'].shape

    new_data = np.empty(shape)
    new_data[:] = np.nan
    new_data[indices[0],indices[1]] = labels
    new_data = xr.DataArray(new_data, dims = dims, coords = coords, name = labelkey)

    return new_data.to_dataset()
 
def get_georef(filename):

    pass


def write_NETCDF(data, filename):
    """
    writes xarray dataset to NETCDF file
    """
    data.to_netcdf(path=filename, mode='w')


def check_nwcsaf_version(labels):
    """
    checks if cloud type labels are mapped by the 2013-netcdf standard

    uses occurences in layers 16-19 (only in use at 2013 standard)
    and 7,9,11,13 (only in use at 2016 standard) 
    """

    high_sum = odd_sum = 0
    for i in range(16,20):
        high_sum += (labels == i).sum()
    for i in range(7,14,2):
        odd_sum = (labels == i).sum()

    if (high_sum > 0 and odd_sum == 0):
        return 'v2013'
    if (high_sum == 0 and odd_sum > 0):
        return 'v2018'
    return None

    """
    maps netcdf cloud types from the 2013 standard to the 2016 standard
    """
def switch_nwcsaf_version(labels, target_version):
    if (target_version == 'v2018'):
        return switch_2016(labels)
    if (target_version == 'v2013'):
        return switch_2013(labels)


def switch_2016(labels):
    """
    maps netcdf cloud types from the 2013 standard to the 2016 standard
    """
    labels[labels == 6.0] = 5.0 # very low clouds
    labels[labels == 8.0] = 6.0 # low clouds
    labels[labels == 10.0] = 7.0 # middle clouds
    labels[labels == 12.0] = 8.0 # high opaque clouds
    labels[labels == 14.0] = 9.0 # very high opaque clouds
    labels[labels == 19.0] = 10.0 # fractional clouds
    labels[labels == 15.0] = 11.0 # high semitransparent thin clouds
    labels[labels == 16.0] = 12.0 # high semitransparent moderatly thick clouds
    labels[labels == 17.0] = 13.0 # high semitransparent thick clouds
    labels[labels == 18.0] = 14.0 # high semitransparent above low or medium clouds
    # missing: 15:  High semitransparent above snow/ice
    return labels



def switch_2013(labels):
    """
    maps netcdf cloud types from the 2016 standard to the 2013 standard
    """
    labels[labels == 15.0] = 18.0 # high semitransparent above snow/ice
    labels[labels == 14.0] = 18.0 # high semitransparent above low or medium clouds
    labels[labels == 13.0] = 17.0 # high semitransparent thick clouds
    labels[labels == 12.0] = 16.0 # high semitransparent moderatly thick clouds
    labels[labels == 11.0] = 15.0 # high semitransparent thin clouds
    labels[labels == 10.0] = 19.0 # fractional clouds
    labels[labels == 9.0] = 14.0 # very high opaque clouds
    labels[labels == 8.0] = 12.0 # high opaque clouds
    labels[labels == 7.0] = 10.0 # middle clouds
    labels[labels == 6.0] = 8.0 # low clouds
    labels[labels == 5.0] = 6.0 # very low clouds

    return labels