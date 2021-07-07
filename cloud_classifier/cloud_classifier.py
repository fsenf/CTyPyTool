import json
import os
import numpy as np

import cloud_trainer as ct
import data_handler as dh
import base_class

import importlib

importlib.reload(ct)
importlib.reload(dh)
importlib.reload(base_class)



class cloud_classifier(base_class.base_class):
    """
    
    bla 

    """


    def __init__ (self, **kwargs):
        self.data_handler = dh.data_handler()
        self.cloud_trainer = ct.cloud_trainer()
        #cloud classifier manages all parameters of member classes
        class_variables = self.data_handler.get_class_variables() + self.cloud_trainer.get_class_variables()
        super().__init__(class_variables, **kwargs)




    def create_new_classifer(self, name, path = None):
        """
        Creates a persistant classifier project.


        Parameters
        ----------
        name : string
            Name of the the project that will be created

        path : string (Optional)
            Path to the directory where the project will be stored. If none is given, 
            current directory will be used.
        """

        if (path is None):
            path = os.getcwd()

        folder = os.path.join(path, name)
        
        try:
            os.mkdir(folder)
        except Exception:
            print("Could not create classifier project at given location")
            return

        config_file = os.path.join(folder, "config.json")
        self.save_parameters(config_file)




    def generate_filelist_from_folder(self, path):
        """
        Extracts trainig files from folder
        Reads all matching files of satellite and label data from folder and adds them to project

        Parameters
        ----------
        path : string 
            Path to the folder containig the data files
        """  
        ################TODO#####################
        pass 


    def add_training_files(self, filename_data, filename_labels):
        self.data_handler.add_training_files(filename_data, filename_labels)

    

    def load_classifier(self, path):
        """
        Loads a persistant classifier project.

        Parameters
        ----------
        path : string 
            Path to the directory where the clasifier is stored.
        """  

        config_file = os.path.join(path, "config.json")
        self.load_parameters(config_file)



