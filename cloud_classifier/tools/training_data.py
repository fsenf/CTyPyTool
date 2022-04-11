import xarray as xr
import random
import numpy as np
import time
import tools.nwcsaf_tools as nwc


"""
Adds helper-methods for extracting machine learining data from NETCDF datasets
"""


def sample_training_sets(training_sets, n, hours, indices, input_channels,  ct_channel = "CT", verbose = False):
    start = time.time()
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

    print("sampling took " + str(time.time()-start) + " seconds")
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


def clean_training_set(vectors, labels, verbose = True):
    """
    Remove vectors and corresponding labels containing nans
    
    """
    valid = ~np.isnan(vectors).any(axis=1)
    valid_l = ~np.isnan(labels)
    valid = np.logical_and(valid, valid_l)
    d = valid.size - vectors[valid].shape[0]
    if (d>0 and verbose):
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



def extract_labels(filename, indices = None, hour = 0, ct_channel = "CT"):
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



def translate_mergeList(merge_list):
    _, indices, labels = nwc.definde_NWCSAF_variables()
    intLabel = [int(i) for i in indices]
    merge_ints = []
    for merge_pair in merge_list:
        l1 = labels.index(merge_pair[0]) 
        l2 = labels.index(merge_pair[1])
        merge_ints.append([intLabel[l1],intLabel[l2]])
    return merge_ints


def merge_labels(labels, merge_list):
    if (merge_list):
        ĺabels = np.array(labels)
        merge_ints = translate_mergeList(merge_list)
        for merge_pair in merge_ints:
            labels[labels == merge_pair[0]] = merge_pair[1]

