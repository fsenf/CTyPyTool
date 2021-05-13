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
        self.netcdf_in_version = 'auto' # other values: 'nc2013' , 'nc2016'
        self.netcdf_out_version = 'nc2016' # other values: 'nc2013'


    def set_input_channels(self, input_channels = ['bt062', 'bt073', 'bt087', 'bt097', 'bt108', 'bt120', 'bt134']):
        """
        Sets the channels used for the data extraction.

        Parameters
        ----------
        input_channels : list of strings
            (Optional) Names of the input channels to be used from the satelite data
            Default is: ['bt062', 'bt073', 'bt087', 'bt097', 'bt108', 'bt120', 'bt134']
        """
        self.input_channels = input_channels



    def set_netcdf_version(self, in_version = 'nc2016', out_version = 'nc2016'):
        """
        Specifies if cloud type mapping follows old or new definitions

        Parameters
        ----------
        in_version : string 
            (Optional) netcdf-Version of input data. Options are 'auto', 'nc2013', 'nc2016'

        out_version  string
            (Optional) netcdf-Version of the output data.  Options are 'nc2013', 'nc2016'

        """
        if (not (in_version == 'auto' or in_version == 'nc2013' or in_version == 'nc2016')):
            print("NETCDF-in-version must be specified as 'nc2013', 'nc2016' or 'auto' ")
            return
        self.netcdf_in_version = in_version
        if (not ( out_version == 'nc2013' or out_version == 'nc2016')):
            print("NETCDF-out-version must be specified as 'nc2013' or 'nc2016'  ")
            return
        self.netcdf_out_version = out_version


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
        m = xr.DataArray([row for row in mask_data[selected_mask]], name = selected_mask)
        self.masked_indices = np.where(m)

        # for key in mask_data.keys():
        #     if key == "_source":
        #         continue
        #     m = xr.DataArray([row for row in mask_data[key]], name = key)
        #     mask_xr[key] = m
        # self.masked_indices = np.where(mask_xr[selected_mask])
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
            List of tuples containing the filenames for training data and corresponding labels
            Is requiered if no training sets have been added to the data_handler object

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
        vectors, labels = ex.sample_training_sets(training_sets, self.n, self.hours, self.masked_indices, 
                                                self.input_channels)

        # Remove nan values
        vectors, labels = ex.clean_training_set(vectors, labels)

        if (self.cDV):
            # create difference vectors
            vectors = ex.create_difference_vectors(vectors, self.kOV)
        
        if (self.netcdf_in_version == 'auto'):
            self.check_netcdf_version(labels, True)

        if (self.netcdf_in_version is not self.netcdf_out_version):
            labels = ex.switch_netcdf_version(labels, self.netcdf_out_version)

        return vectors, labels


    def check_netcdf_version(self, labels, set_value = False):
        """
        Checks if a set of labels follows the 2013 or 2016 standard.


        Parameters
        ----------
        labels : array like
            Array of labels

        set_value : bool
            If true the flag for the netcdf version of the input data is set accordingly
        
        Returns
        -------
        string or None
            String naming the used version or None if version couldnt be determined
        """
        r = ex.check_netcdf_version(labels)
        if (r is None):
            print("Could not determine netcdf version of the labels")
        else:
            if (r == "nc2016"):
                print("The cloud type data is coded after the new (2016) standard")
            if (r == "nc2013"):
                print("The cloud type data is coded after the old (2013) standard")
            if (set_value):
                print("NETCDF-in-version set accordingly")
                self.netcdf_in_version = r
        return r



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






    def exctract_labels(filename, indices, hour = 0,):
        """
        Extract labels from netCDF file at given indices and time

        Parameters
        ----------
        filename_data : string
            Filename of the label data
        
        indices : tuple of arrays
            tuple of int arrays specifing the indices of the returned labels

        hour : int
            0-23, hour of the day at which the data set is read

        Returns
        -------
        numpy array 
            labels at the specified indices and time

        """ 
        labels = ex.exctract_labels(filename, indices, hour)

        if (self.netcdf_in_version == 'auto'):
            self.check_netcdf_version(labels, True)

        if (self.netcdf_in_version is not self.netcdf_out_version):
            labels = ex.switch_netcdf_version(labels, self.netcdf_out_version)

        return labels


    def make_xrData(self, labels, indices, reference_filename = None, NETCDF_out = None):
        """
        Transforms a set of predicted labels into xarray-dataset  

        Parameters
        ----------
        labels : array-like
            int array of label data

        indices : tuple of array-like
            tuple of int arrays specifing the indices of the given labels in respect to the
            ccordiantes from a reference file  

        reference_filename : string
            (Optional) filename of a NETCDF file with the same scope as the label data
            Is requiered if no training sets have benn added to the data_handler
        
        NETCDF_out : string
            (Optional) If specified, the labels will be written to a NETCDF file with this name

        Returns
        -------
        xarray dataset 
            labels in the form of an xarray dataset

        """

        if (reference_filename is None):
            if (self.training_sets is None):
                print("No refrence file given!")
                return
            else:
                reference_filename = self.training_sets[0][0]
                print("Using training data as reference!")

        xar = ex.make_xarray(labels, indices, reference_filename)

        if (not NETCDF_out is None):
            ex.write_NETCDF(xar, NETCDF_out)

        return xar



    def save_training_set(self, filename, vectors, labels):
        """
        Saves a set of training vectors and labels

        Parameters
        ----------
        filename : string
            Name of the file into which the vector set is saved.
        """
        dump([vectors, labels], filename)


    def load_training_set(self, filename):
        """
        Loads a set of training vectors and labels
        
        Parameters
        ----------
        filename : string
            Name if the file the vector set is loaded from.

        Returns
        -------
        tuple containing a set of training vectors and corresponing labels
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