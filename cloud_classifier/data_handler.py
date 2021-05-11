import numpy as np
import xarray as xr
import random
import h5py 
import tools.data_extraction as ex
from joblib import dump, load

import importlib
importlib.reload(ex)

class data_handler:

    """
    Class that faciltates data extraction and processing for the use in machine learning from NETCDF-satelite data.
    """



    def __init__(self):
        self.cDV = False
        self.kOV = False
        self.n = 1000
        self.hours=range(24)


        self.masked_indices = None
        self.training_sets = []

        # default input channels
        self.input_channels = ['bt062', 'bt073', 'bt087', 'bt097', 'bt108', 'bt120', 'bt134'] 


    def set_input_channels(self, input_channels):
        """
        Sets the paramerters for the data extraction.

        Parameters
        ----------
        n : list 
            Names of the input channels to be used from the satelite data
        """
        self.input_channels = input_channels



    def set_extraction_parameters(self, n=1000, hours=range(24), cDV=True, kOV=True):
        """
        Sets the paramerters for the data extraction.

        Parameters
        ----------
        n : int
            (Optional) Number of samples taken for each time value for each training set

        hour : list
            (Optional) Hours from which vectors are cerated

        cDV: bool
            (Optional) Calculate inter-value-difference-vectors for use as training vectors. Default True.

        kOV : bool
            (Optional) When using difference vectors, also keep original absolut values as 
            first entries in vector. Default True.
        """
        self.cDV = cDV
        self.kOV = kOV
        self.n = n
        self.hours=hours

        

    def set_indices_from_mask(self, filename, selected_mask):
        """
        Sets indices according to a selected mask

        Reads mask-data from h5 file and converts it into xr.array. From this data the indices corresponding
        to the selected mask are extracted and saved

        Parameters
        ----------
        filename : string
            Filename of the mask-data
            
        selected_mask : string
            Key of mask to be used

        """
        mask_data = h5py.File(filename, 'r')
        mask_xr = xr.Dataset()

        for key in mask_data.keys():
            if key == "_source":
                continue
            m = xr.DataArray([row for row in mask_data[key]], name = key)
            mask_xr[key] = m
        self.masked_indices = np.where(mask_xr[selected_mask])
        return self.masked_indices



    def add_training_files(self, filename_data, filename_labels):
        """
        Takes filenames of satelite data and according labels and adds it to the data_handler
        
        Parameters
        ----------
        filename_data : string
            Filename of the sattelite data
            
        filename_labels : string
            Filename of the label dataset

        """
        self.training_sets.append([filename_data, filename_labels])
        return



    def create_training_set(self, training_sets = None):
        """
        Creates a set of training vectors from NETCDF datasets

        Samples a set of satellite data and corresponding labels at n random positions 
        for each hour specified. 


        Parameters
        ----------
        training_sets (Optional) : list of string tuples
            Each tuple contains the filenames for a set of satellilte data and corresponding labels
            If none are given, the method uses those trainig sets added previously to this instance of data_handler

        Returns
        -------
        tuple of numpy arrays
            Arrays containig the training vectors and corresponding labels

        """
        if (training_sets is None):
            training_sets = self.training_sets
        if(training_sets is None):
            print("No training data added.")
            return

        # Get vectors from all added training sets
        vectors, labels = ex.sample_training_sets(self.training_sets, self.n, self.hours, 
                                                self.masked_indices, self.input_channels)

        # Remove nan values
        vectors, labels = ex.clean_training_set(vectors, labels)

        if (self.cDV):
            # create difference vectors
            vectors = ex.create_difference_vectors(vectors, self.kOV)

        return vectors, labels


    def create_test_vectors(self, filename_data, hour=0):
        """
        Extracts feature vectors from given NETCDF file at a certain hour.


        Parameters
        ----------
        filename_data : string
            Filename of the sattelite data

        hour : int
            0-23, hour of the day at which the data set is read

        Returns
        -------
        tuple of numpy arrays
            Array containig the test vectors and another array containing the indices those vectors belong to

        """
        sat_data = xr.open_dataset(filename_data)
        indices = self.masked_indices
        if (indices is None):
            # get all non-nan indices from the first layer specified in input channels
            indices = np.where(~np.isnan(sat_data[input_channels[0]][0]))
            print("No mask indices given, using complete data set")

        vectors = ex.extract_feature_vectors(sat_data, indices, hour, self.input_channels)
        vectors, indices = ex.clean_test_vectors(vectors, indices)
        if (self.cDV):
            vectors = ex.create_difference_vectors(vectors, self.kOV)

        return vectors, indices






    def exctract_labels(filename, indices, hour = 0):
        """
        Extract labels from netCDF file at given indices and time

        Parameters
        ----------
        filename_data : string
            Filename of the label data
        
        indices : tuple of arrays
            indices of labels that 
        hour : int
            0-23, hour of the day at which the data set is read

        Returns
        -------
        tuple of numpy arrays
            Array containig the test vectors and another array containing the indices those vectors belong to

        """
        return ex.exctract_labels(xr.open_dataset(filename), indices, hour)




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



    def save_training_set(self, filename, vectors, labels):
        """
        Saves a set of already created training vectors and labels

        Parameters
        ----------
        filename : string
            Name of the file into which the vector set is saved.
        """
        dump([vectors, labels], filename)


    def load_training_set(self, filename):
        """
        Loads a set of already created training vectors and labels
        
        Parameters
        ----------
        filename : string
            Name if the file the vector set is loaded from.
        """
        v, l = load(filename)
        return v,l

























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