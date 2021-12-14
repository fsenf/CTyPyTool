import numpy as np
import xarray as xr
import random
import h5py
import re
import os
from joblib import dump, load
import warnings


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cartopy
import cartopy.crs as ccrs

import tools.training_data as td
import tools.file_handling as fh
import base_class
import tools.confusion as conf

#
import importlib
importlib.reload(td)
importlib.reload(fh)

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
            'reference_file',
            'georef_file',
            'merge_list'
            ]

        super().init_class_variables(class_variables)
        super().__init__( **kwargs)
        self.masked_indices = None

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
        vectors, labels = td.sample_training_sets(training_sets, self.samples, self.hours, masked_indices, 
                                                self.input_channels, self.cloudtype_channel, 
                                                verbose = self.verbose)

        # Remove nan values
        vectors, labels = td.clean_training_set(vectors, labels)

        if (self.difference_vectors):
            # create difference vectors
            vectors = td.create_difference_vectors(vectors, self.original_values)
        
        if (self.nwcsaf_in_version == 'auto'):
            self.nwcsaf_in_version = self.check_nwcsaf_version(labels, verbose = False)
        
        labels = fh.switch_nwcsaf_version(labels, self.nwcsaf_out_version, self.nwcsaf_in_version)

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
            labels = td.extract_labels(filename = filename, ct_channel = self.cloudtype_channel)
            
        r = fh.check_nwcsaf_version(labels)
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

        vectors = td.extract_feature_vectors(sat_data, indices, hour, self.input_channels)
        vectors, indices = td.clean_test_vectors(vectors, indices)
        if (self.difference_vectors):
            vectors = td.create_difference_vectors(vectors, self.original_values)

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

        labels = td.extract_labels(filename, indices, hour, self.cloudtype_channel)

        if (self.nwcsaf_in_version == 'auto'):
            self.check_nwcsaf_version(labels, True)

        if (self.nwcsaf_in_version is not self.nwcsaf_out_version):
            labels = fh.switch_nwcsaf_version(labels, self.nwcsaf_out_version)

        return labels


    def make_xrData(self, labels, indices, reference_file = None, NETCDF_out = None, 
            prob_data = None):
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

        out = xr.open_dataset(reference_file)

        shape = out[self.cloudtype_channel][0].shape # 0 being the hour
        new_data = np.empty(shape)
        new_data[:] = np.nan
        new_data[indices[0],indices[1]] = labels
        out[self.cloudtype_channel][0] = new_data


        if(prob_data is not None):
            shape += (len(prob_data[0]), )
            new_data = np.empty(shape)
            new_data[:] = np.nan
            new_data[indices[0],indices[1]] = prob_data

            new_dims = out[self.cloudtype_channel][0].dims
            new_dims += ("labels",)
            out["label_probability"] = (new_dims, new_data)

        if (not NETCDF_out is None):
            fh.write_NETCDF(out, NETCDF_out)

        return out

 


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
        for key in data.keys():
            if(not key == self.cloudtype_channel):
                data =data.drop(key)

        data.to_netcdf(path=output_file, mode ='w')
        self.reference_file = output_file






################ Ploptting, move to other file







    def plot_data(self, label_file, reduce_to_mask = True , extent= None,  cmap = "hot", mode = "label", 
        subplot = False, pos = None, colorbar = False, cb_pos = 0.95):

        if (mode == "label"):
            ct_colors, ct_indices, ct_labels = fh.definde_NWCSAF_variables()
            cmap = plt.matplotlib.colors.ListedColormap(ct_colors)
            vmin = 1
            vmax = 15
        elif (mode == "proba"):
            vmin = 0.0
            vmax = 1.0

        if (not subplot):
            plt.figure(figsize=(13, 8))
        if(extent is None):
             extent = [-6, 42, 25, 50]

        if(subplot):
            ax = plt.subplot(pos[0], pos[1], pos[2], projection=ccrs.PlateCarree())
        else:
            extent = [-6, 42, 25, 50]
            ax = plt.axes(projection=ccrs.PlateCarree())

        ax.coastlines(resolution='50m')
        ax.set_extent(extent)
        ax.add_feature(cartopy.feature.LAND, edgecolor='black')
        data, x, y = self.get_plotable_data(data_file = label_file, reduce_to_mask = reduce_to_mask, mode = mode)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pcm = ax.pcolormesh(x,y, data, cmap = cmap, vmin = vmin, vmax = vmax)

        if(colorbar):
            fig = plt.gcf()
            a2 = fig.add_axes( [cb_pos, 0.22, 0.015, 0.6]) 
            cbar = plt.colorbar(pcm, a2)
            if (mode == "label"):
                cbar.set_ticks(ct_indices)
                cbar.ax.tick_params(labelsize=14)
                cbar.set_ticklabels(ct_labels)

        if(subplot):
            return ax, data


    def plot_probas(self, label_file, truth_file = None, georef_file = None, reduce_to_mask = True, 
        plot_corr = False, plot_titles = None, hour = None, save_file = None, show = True):

        gt = (not truth_file is None)
        extent = [-6, 42, 25, 50]
        length = 2
        if(not truth_file is None):
            length+=1
        if(plot_corr):
            length+=1

        fig = plt.figure(figsize=(length*13, 8))
        fig.patch.set_alpha(1)

        if (gt):
            pos = [1,length,length]
            ax, truth = self.plot_data(truth_file, reduce_to_mask, pos = pos, subplot = True, colorbar = True)
            if (hour is not None):
                text = "Time: {:02d}:00".format(hour)
                ax.text(10, 22, text, fontsize = 16)
            if (plot_titles is None or length > len(plot_titles)):
                ax.set_title("Ground Truth", fontsize = 20)

        modes = ["proba", "label"]
        cb = [True, not gt]
        cb_p = [0.05, 0.95]
        for i in range(len(modes)):
            pos = [1,length,i+1]
            ax,data = self.plot_data(label_file, reduce_to_mask = reduce_to_mask, pos = pos, subplot = True, mode = modes[i], colorbar = cb[i], cb_pos = cb_p[i])

            if (not plot_titles is None):
                ax.set_title(plot_titles[i], fontsize = 20)

            if (gt and i == 1):
                text = fh.get_match_string(data, truth)
                ax.text(10, 22, text, fontsize = 16)

        plt.subplots_adjust(wspace=0.05)
        if(save_file is not None):
            plt.savefig(save_file, transparent=False)
        if(show):
            plt.show()
        plt.close()


    def plot_multiple(self, label_files, truth_file = None, georef_file = None, 
            reduce_to_mask = True, plot_titles = None, hour = None, save_file = None, show = True):

        extent = [-6, 42, 25, 50]

        length = len(label_files)
        lab_lenght = length 
        if(not truth_file is None):
            length+=1
        fig = plt.figure(figsize=(length*13, 8))
        fig.patch.set_alpha(1)

        # plot ground truth
        if (not truth_file is None):
            pos = [1,length,length]
            ax, truth = self.plot_data(truth_file, reduce_to_mask = reduce_to_mask, pos = pos, subplot = True, colorbar = True)
            if (hour is not None):
                text = "Time: {:02d}:00".format(hour)
                ax.text(10, 22, text, fontsize = 16)
            if (length > len(plot_titles)):
                ax.set_title("Ground Truth", fontsize = 20)

        # plot labels
        for i in range(lab_lenght):
            pos = [1,length,i+1]
            ax,data = self.plot_data(label_files[i], reduce_to_mask, pos = pos, subplot = True)

            if (not plot_titles is None and i < len(plot_titles)):
                ax.set_title(plot_titles[i], fontsize = 20)

            if (not truth_file is None and i < lab_lenght):
                text = fh.get_match_string(data, truth)
                ax.text(10, 22, text, fontsize = 16)

        plt.subplots_adjust(wspace=0.05)
        if(save_file is not None):
            plt.savefig(save_file, transparent=False)
        if(show):
            plt.show()
        plt.close()

    def get_plotable_data(self, input_data = None, data_file = None, georef_file = None, 
            reduce_to_mask = True, get_coords = True, mode = "label"):
        """
        Returns input_data and coordinates in a plottable format 
        """

        if(input_data is None and data_file is None):
            raise ValueError("No input data given!")
    
        elif (input_data is None):
            input_data = xr.open_dataset(data_file)

        if (mode == "label"):
            data = input_data[self.cloudtype_channel][0]
        elif (mode == "proba"):
            data = np.amax(input_data["label_probability"], axis = 2)

        # shrink to area, transform to numpy
        indices = None
        if(reduce_to_mask):
            if self.masked_indices is None:
                self.set_indices_from_mask(self.mask_file, self.mask_key)
            indices = self.masked_indices
        else:
            indices = np.where(~np.isnan(data))

        out_data = np.empty(data.shape)
        out_data[:] = np.nan
        data = np.array(data)[indices[0], indices[1]]
        out_data[indices[0],indices[1]] = data
        out_data = fh.switch_nwcsaf_version(out_data, target_version = 'v2018')
            
        if(not get_coords):
            return out_data
        else:
            if(georef_file is None):
                georef_file = self.georef_file
            if(georef_file is None):
                x = input_data.coords['lon']
                y = input_data.coords['lat']
            else:
                georef = xr.open_dataset(georef_file)
                x = georef.coords['lon']
                y = georef.coords['lat']

            return out_data, x, y



    def plot_coocurrence_matrix(self, label_file, truth_file, normalize=True):
        label_data = self.get_plotable_data(data_file = label_file, reduce_to_mask = True, get_coords = False)
        truth_data = self.get_plotable_data(data_file = truth_file, reduce_to_mask = True, get_coords = False)
        conf.plot_coocurrence_matrix(label_data, truth_data, normalize=normalize)
