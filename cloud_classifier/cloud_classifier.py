import json
import os
import numpy as np
import shutil


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


class cloud_classifier(cloud_trainer, data_handler):
    """
    
    bla PIPELINE

    """

    def __init__ (self, **kwargs):
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
        self.load_all(self.project_path)

    def save_project(self):
        self.save_all(self.project_path)
    
    def set_project_path(self, path):
        self.project_path = path





    #############           Function Overwrites         ######################
    ##########################################################################



    def generate_filelist_from_folder(self, folder = None, additive = False, verbose = False):
        super().generate_filelist_from_folder(additive = False)
        self.save_parameters(path = self.project_path, type = "training_data")
        if (verbose):
            print("Filelist created!")


    def set_indices_from_mask(self, verbose = False):
        super().set_indices_from_mask(self.mask_file, self.mask_key)
        if (verbose):
            print("Masked indices created!")


    def create_training_set(self, verbose = False):
        v,l = super().create_training_set()
        filename = os.path.join(self.project_path, "data", "training_data")
        self.save_training_set(v,l, filename)
        if (verbose):
            print("Training data created!")






    def train_classifier(self, verbose = True):
        """
        
        """
        self.generate_filelist_from_folder(additive = False, verbose = verbose)
        self.set_indices_from_mask(verbose = verbose)
        self.create_training_set(verbose = verbose)