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
            "input_source_folder"
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





    #############           Function Overwrites         ######################
    ##########################################################################



    def extract_training_filelist(self, verbose = True):
        self.training_sets =  self.generate_filelist_from_folder(self.data_source_folder)
        filepath = os.path.join(self.project_path, "settings", "training_sets.json")
        self.save_parameters(filepath)
        if (verbose):
            print("Filelist created!")


    def set_indices_from_mask(self, verbose = True):
        super().set_indices_from_mask(self.mask_file, self.mask_key)
        if (verbose):
            print("Masked indices created!")


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


    ##########################################################################
    def run_training_pipeline(self, verbose = True):
        """
        
        """
        self.extract_training_filelist(verbose = verbose)
        self.set_indices_from_mask(verbose = verbose)
        v,l = self.create_training_set(verbose = verbose)
        self.train_classifier(v,l)


    def run_prediction_pipeline(self, verbose = True, reload_all = True):
        if(self.classifier is None or reload_all):
            filename = os.path.join(self.project_path, "data", "classifier")
            self.load_classifier(filename)
            if (verbose):
                "Classifier loaded"







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
            folder = self.data_source_folder
        if (folder is None):
            print("No folder specified!")
            return

        if ("TIMESTAMP" not in self.sat_file_structure or 
            "TIMESTAMP" not in self.label_file_structure):
            print ("Specified file structure must contain region marked as 'TIMESTAMP'")
            return

        pattern = "(.{" + str(self.timestamp_length) + "})"

        sat_pattern = self.sat_file_structure.replace("TIMESTAMP", pattern)
        lab_pattern = self.label_file_structure.replace("TIMESTAMP", pattern)

        sat_comp = re.compile(sat_pattern)
        lab_comp = re.compile(lab_pattern)
        sat_files, lab_files = {}, {}

        files = os.listdir(folder)
        for file in files:
            sat_id = sat_comp.match(file)
            lab_id = lab_comp.match(file)
            
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
