import numpy as np
import xarray as xr
import random
import h5py
import re
import os
from joblib import dump, load


import tools.data_extraction as ex
import tools.plotting as pl
import base_class
#
import importlib
importlib.reload(ex)
importlib.reload(pl)

from base_class import base_class
from cloud_trainer import cloud_trainer
   


class data_handler(base_class):

    """
    Class that faciltates data extraction and processing for the use in machine learning from NETCDF-satelite data.
    """



    def __init__(self, **kwargs):

        #self.set_default_parameters(reset_data = True)
        class_variables = [
            "data_source_folder", 
            "timestamp_length",
            "sat_file_structure",
            "label_file_structure",
            'difference_vectors', 
            'original_values', 
            'samples', 
            'hours', 
            'input_channels',
            'cloudtype_channel',
            'nwcsaf_in_version',
            'nwcsaf_out_version',
            'verbose',
            'training_sets',
            'mask_file',
            'mask_key',
            'mask_sea_coding',
            'reference_file'
            ]

        super().init_class_variables(class_variables)
        super().__init__( **kwargs)
        self.masked_indices = None
        self.latest_test_file = None

    def set_indices_from_mask(self, filename, selected_mask):
        """
        Sets indices according to a selected mask

        Reads mask-data from h5 file and converts it into xr.array. From this data the indices corresponding
        to the selected mask are extracted and saved

            
        selected_mask : string
            Key of mask to be used

        """
        mask_data = h5py.File(filename, 'r')
        m = xr.DataArray([row for row in mask_data[selected_mask]], name = selected_mask)
        self.masked_indices = np.where(m == self.mask_sea_coding)

        self.mask = [filename, selected_mask]
        return self.masked_indices



    def add_training_files(self, filename_data, filename_labels):
        """
        Takes filenames of satelite data and according labels and adds it to the data_handler.
        
        Parameters
        ----------
        filename_data : string
            Filename of the sattelite data
            
        filename_labels : string
            Filename of the label dataset

        """
        if (self.training_sets is None):
            self.training_sets = []
        self.training_sets.append([filename_data, filename_labels])
        if reference_file is None:
            self.reference_file = filename_labels
        return



    def create_training_set(self, training_sets = None, masked_indices = None):
        """
        Creates a set of training vectors from NETCDF datasets.

        Samples a set of satellite data and corresponding labels at samples random positions 
        for each hour specified. 


        Parameters
        ----------
        training_sets (Optional) : list of string tuples
            List of tuples containing the filenames for training data and corresponding labels
            Is requiered if no training sets have been added to the data_handler object
        masked_indices (Optional) : numpy array

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

        if (masked_indices is None):
            masked_indices = self.masked_indices
        # Get vectors from all added training sets
        vectors, labels = ex.sample_training_sets(training_sets, self.samples, self.hours, masked_indices, 
                                                self.input_channels, self.cloudtype_channel, 
                                                verbose = self.verbose)

        # Remove nan values
        vectors, labels = ex.clean_training_set(vectors, labels)

        if (self.difference_vectors):
            # create difference vectors
            vectors = ex.create_difference_vectors(vectors, self.original_values)
        
        if (self.nwcsaf_in_version == 'auto'):
            self.nwcsaf_in_version = self.check_nwcsaf_version(labels, verbose = False)
        
        labels = ex.switch_nwcsaf_version(labels, self.nwcsaf_out_version, self.nwcsaf_in_version)

        return vectors, labels




    def check_nwcsaf_version(self, labels = None, filename = None, verbose = True):
        """
        Checks if a set of labels follows the 2013 or 2016 standard.


        Parameters
        ----------
        labels : array like
            Array of labels

        set_value : bool
            If true the flag for the ncwsaf version of the input data is set accordingly
        
        Returns
        -------
        string or None
            String naming the used version or None if version couldnt be determined
        """
        if (labels is None and filename is None):
            raise  ValueError("Label or filename must be specified")
        if (labels is None):
            labels = ex.extract_labels(filename = filename, ct_channel = self.cloudtype_channel)
            
        r = ex.check_nwcsaf_version(labels)
        if(verbose):
            if (r is None):
                print("Could not determine ncwsaf version of the labels")
            else:
                if (r == "v2018"):
                    print("The cloud type data is coded after the new (2018) standard")
                if (r == "v2013"):
                    print("The cloud type data is coded after the old (2013) standard")
        return r



    def create_test_vectors(self, filename, hour=0):
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
        indices = self.masked_indices
        if (indices is None):
            # get all non-nan indices from the first layer specified in input channels
            indices = np.where(~np.isnan(sat_data[self.input_channels[0]][0]))
            print("No mask indices given, using complete data set")

        vectors = ex.extract_feature_vectors(sat_data, indices, hour, self.input_channels)
        vectors, indices = ex.clean_test_vectors(vectors, indices)
        if (self.difference_vectors):
            vectors = ex.create_difference_vectors(vectors, self.original_values)

        self.latest_test_file = filename
        return vectors, indices




    def extract_labels(self, filename, indices = None, hour = 0):
        """
        Extract labels from netCDF file at given indices and time

        Parameters
        ----------
        filename : string
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
        if (indices is None):
            # get all non-nan indices from the first layer specified in input channels
            if (not self.masked_indices is None):
                indices = self.masked_indices
                print("No indices specified, using mask indices")

            else:
                print("No mask indices given, using complete data set")

        labels = ex.extract_labels(filename, indices, hour, self.cloudtype_channel)

        if (self.nwcsaf_in_version == 'auto'):
            self.check_nwcsaf_version(labels, True)

        if (self.nwcsaf_in_version is not self.nwcsaf_out_version):
            labels = ex.switch_nwcsaf_version(labels, self.nwcsaf_out_version)

        return labels


    def make_xrData(self, labels, indices, reference_file = None, NETCDF_out = None):
        """
        Transforms a set of predicted labels into xarray-dataset  

        Parameters
        ----------
        labels : array-like
            Int array of label data

        indices : tuple of array-like
            Indices of the given labels in respect to the coordiantes from a reference file  

        reference_file : string
            (Optional) filename of a NETCDF file with the same scope as the label data.
            This field is requiered if no refernce file has been created before!
        
        NETCDF_out : string
            (Optional) If specified, the labels will be written to a NETCDF file with this name

        Returns
        -------
        xarray dataset 
            labels in the form of an xarray dataset

        """
        if (reference_file is None and self.reference_file is None):
            raise ValueError("Reference file must be set or specified!")
        if (reference_file is None):
            reference_file = self.reference_file

        xar = ex.make_xrData(labels, indices, reference_file,
                ct_channel = self.cloudtype_channel)

        if (not NETCDF_out is None):
            ex.write_NETCDF(xar, NETCDF_out)

        return xar



    def save_filelist(self, filename):
        """

        """
        dump(self.training_sets, filename)


    def save_mask_indices(self, filename):
        dump(self.masked_indices, filename)




    def save_training_set(self, vectors, labels, filename):
        """
        Saves a set of training vectors and labels

        Parameters
        ----------
        filename : string
            Name of the file into which the vector set is saved.

        vectors : array like
            The feature vectors of the training set.

        labels : array like
            The labels of the training set
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
        Tuple containing a set of training vectors and corresponing labels
        """
        v, l = load(filename)
        return v,l


    def create_reference_file(self, input_file, output_file):
        """
        Reads a label file and creates a reference file with all meta data in order
        to later use this as a template for writing predicted labels to file.
        """
        data = xr.open_dataset(input_file)
        data.to_netcdf(path=output_file, mode='w')
        self.reference_file = output_file


    def plot_labels(self, data = None, data_file = None, georef_file = None, reduce_to_mask = False):
        """ 
         Plots labels either from a xarray dataset NetCDF file 
        """

        if(data is None and data_file is None):
            print("No data given!")
            return
        elif (data is None):
            data = xr.open_dataset(data_file)
        label_data = data[self.cloudtype_channel][0]
        if(georef_file is None):
            x = data.coords['lon']
            y = data.coords['lat']
        else:
            georef = xr.open_dataset(georef_file)
            x = georef.coords['lon']
            y = georef.coords['lat']

        # shrink to area, transform to numpy
        indices = None
        if(reduce_to_mask):
            if self.masked_indices is None:
                self.set_indices_from_mask(self.mask_file, self.mask_key)
            indices = self.masked_indices

        else:
            indices = np.where(~np.isnan(label_data))

        labels = np.empty(label_data.shape)
        labels[:] = np.nan
        # x_red = y_red = labels
        label_data = np.array(label_data)[indices[0], indices[1]]
        labels[indices[0],indices[1]] = label_data
        labels = ex.switch_nwcsaf_version(labels, target_version = 'v2018')

        # x = np.array(x)[indices[0], indices[1]]
        # y = np.array(y)[indices[0], indices[1]]
        # x_red[indices[0],indices[1]] = x
        # y_red[indices[0],indices[1]] = y
        pl.plot_data(labels, x, y)
