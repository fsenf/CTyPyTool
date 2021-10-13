import json
import os
import numpy as np
import shutil
import re


import cloud_trainer as ct
import data_handler as dh
import base_class as bc

import importlib
importlib.reload(ct)
importlib.reload(dh)
importlib.reload(bc)

from cloud_trainer import cloud_trainer
from data_handler import data_handler
from base_class import base_class
from joblib import dump, load


class cloud_classifier(cloud_trainer, data_handler):
    """
    
    bla PIPELINE

    """

    def __init__ (self, **kwargs):
        class_variables =  {
            "input_source_folder",
            "input_files"
            }

        super().init_class_variables(class_variables)
        super().__init__(**kwargs)





    ############# CREATING, LOADING AND SAVING PROJECTS ######################
    ##########################################################################

    def create_new_project(self, name, path = None):
        """
        Creates a persistant classifier project.


        Parameters
        ----------
        name : string
            Name of the the project that will be created

        path : string (Optional)
            Path to the directory where the project will be stored. If none is given, 
            the current working directory will be used.
        """

        if (path is None):
            path = os.getcwd()

        folder = os.path.join(path, name)
        
        try:
            shutil.copytree(self.default_path, folder)

        except Exception:
            print("Could not initalize project settings at given location")
            return 0

        self.set_project_path(folder)


    def load_project(self, path):
        """
        Loads a persistant classifier project.

        Parameters
        ----------
        path : string 
            Path to the stored project
        """  
        self.set_project_path(path)
        self.load_settings(self.project_path)

    def save_project(self):
        self.save_all(self.project_path)
    
    def set_project_path(self, path):
        self.project_path = path





    #############           Steps of the pipeline         ######################
    ##########################################################################


    #### training
    def extract_training_filelist(self, verbose = True):
        self.training_sets =  self.generate_filelist_from_folder(self.data_source_folder)
        filepath = os.path.join(self.project_path, "settings", "training_sets.json")
        self.save_parameters(filepath)
        if (verbose):
            print("Filelist created!")


    def apply_mask(self, verbose = True):
        super().set_indices_from_mask(self.mask_file, self.mask_key)
        #filename = os.path.join(self.project_path, "data", "masked_indices")
        if (verbose):
            print("Masked indices set!")


    def create_training_set(self, verbose = True):
        v,l = super().create_training_set()
        filename = os.path.join(self.project_path, "data", "training_data")
        self.save_training_set(v,l, filename)
        if (verbose):
            print("Training data created!")
        return v,l

    def train_classifier(self, vectors, labels, verbose = True):
        super().train_classifier(vectors, labels)
        filename = os.path.join(self.project_path, "data", "classifier")
        self.save_classifier(filename)
        if (verbose):
            print("Classifier created!")


    def create_reference_file(self, input_file = None, verbose = True):
        if (input_file is None):
            if (self.training_sets is None):
                raise ValueError("No reference file found")
            input_file = self.training_sets[0][1]
        output_file = os.path.join(self.project_path, "data", "label_reference.nc")
        super().create_reference_file(input_file, output_file)




    ### predicting
    def extract_input_filelist(self, verbose = True):
        self.input_files =  self.generate_filelist_from_folder(folder = self.input_source_folder, no_labels = True)
        filepath = os.path.join(self.project_path, "settings", "input_files.json")
        self.save_parameters(filepath)
        if (verbose):
            print("Input filelist created!")


    def load_classifier(self, reload = False, verbose = True):
        if(self.classifier is None or reload):
            filename = os.path.join(self.project_path, "data", "classifier")
            super().load_classifier(filename)
            filename = os.path.join(self.project_path, "data", "masked_indices")
            if(verbose):
                print("Classifier loaded!")

    def create_input_vectors(self, file, verbose = True):
        vectors, indices = super().create_test_vectors(file)
        if(verbose):
                print("Input vectors created!")
        return vectors, indices


    def predict_labels(self, input_vectors, verbose = True):
        labels = super().predict_labels(input_vectors)
        if(verbose):
            print("Predicted Labels!")
        return labels

    def save_labels(self, labels, indices, sat_file, verbose = True):
        name = self.get_label_name(sat_file)
        filepath = os.path.join(self.project_path, "labels", name)
        self.make_xrData(labels, indices, NETCDF_out = filepath)
        if(verbose):
            print("Labels saved as " + name )





    ##########################################################################

    def run_training_pipeline(self, verbose = True, create_filelist = True):
        if (create_filelist):
            self.extract_training_filelist(verbose = verbose)
        if (self.reference_file is None):
            self.create_reference_file()
        self.apply_mask(verbose = verbose)
        v,l = self.create_training_set(verbose = verbose)
        self.train_classifier(v,l, verbose = verbose)


    def run_prediction_pipeline(self, verbose = True, create_filelist = True):
        if (create_filelist):
            self.extract_input_filelist(verbose = verbose)
        self.load_classifier(reload = True, verbose = verbose)
        self.apply_mask(verbose = verbose)
        for file in self.input_files:
            vectors, indices = self.create_input_vectors(file, verbose = verbose)
            labels = self.predict_labels(vectors, verbose = verbose)
            self.save_labels(labels, indices, file, verbose = verbose)

            #TODO: convert and save labels









    ##########################################################################


    def generate_filelist_from_folder(self, folder = None, no_labels = False):
        """
        Extracts trainig files from folder
        Reads all matching files of satellite and label data from folder and adds them to project

        Parameters
        ----------
        folder : string (Optional)
            Path to the folder containig the data files. If none is given path will be read from settings
        additive : bool
            Default is True. If True, files will be read additive, if False old filelists will be overwritten.
        no_labels : bool 
            Default is False. If True, filelist will contain sat data and labels, if False only sat_data files.
        """

        if (folder is None):
            print("No folder specified!")
            return

        # if ("TIMESTAMP" not in self.sat_file_structure or 
        #     "TIMESTAMP" not in self.label_file_structure):
        #     print ("Specified file structure must contain region marked as 'TIMESTAMP'")
        #     return

        # pattern = "(.{" + str(self.timestamp_length) + "})"

        # sat_pattern = self.sat_file_structure.replace("TIMESTAMP", pattern)
        # lab_pattern = self.label_file_structure.replace("TIMESTAMP", pattern)

        sat_pattern = self.get_filename_pattern(self.sat_file_structure, self.timestamp_length)
        lab_pattern = self.get_filename_pattern(self.label_file_structure, self.timestamp_length)
        sat_files, lab_files = {}, {}

        files = os.listdir(folder)
        for file in files:
            sat_id = sat_pattern.match(file)
            lab_id = lab_pattern.match(file)
            
            if (sat_id):
                sat_files[sat_id.group(1)] = os.path.join(folder, file)
            elif (lab_id):
                lab_files[lab_id.group(1)] = os.path.join(folder, file)


        if no_labels:
            # return onky satelite date 
            return list(sat_files.values())
        else:
            # find pairs of sat data and labels
            training_sets = []
            for key in sat_files.keys():
                if(key in lab_files):
                    training_sets.append([sat_files[key], lab_files[key]])
            return  training_sets


    def get_filename_pattern(self, structure, t_length):
        if ("TIMESTAMP" not in structure):
            raise ValueError("Specified file structure must contain region marked as 'TIMESTAMP'")

        replacemnt =  "(.{" + str(t_length) + "})"
        pattern = structure.replace("TIMESTAMP", replacemnt)
        return re.compile(pattern)


    def get_label_name(self, sat_file):
        # get_filename if whole path is given
        name_split = os.path.split(sat_file)
        reference_file = name_split[1]
        # compute regex patterns
        sat_pattern = self.get_filename_pattern(self.sat_file_structure, self.timestamp_length)

        timestamp = sat_pattern.match(reference_file)
        if(not timestamp):
            raise Exception("Satelite data file does not match specified naming pattern")
        timestamp = timestamp.group(1)
        label_file = self.label_file_structure.replace("TIMESTAMP", timestamp)
        name, ext = os.path.splitext(label_file)
        label_file = name + "_predicted" + ext
        return label_file