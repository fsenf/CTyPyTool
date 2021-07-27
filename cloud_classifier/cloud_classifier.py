import json
import os
import numpy as np
import shutil

import cloud_trainer as ct
import data_handler as dh
import base_class

import importlib

importlib.reload(ct)
importlib.reload(dh)
importlib.reload(base_class)



class cloud_classifier(base_class.base_class):
    """
    
    bla PIPELIN

    """


    def __init__ (self, **kwargs):
        self.data_handler = dh.data_handler( **kwargs)
        self.cloud_trainer = ct.cloud_trainer( **kwargs)

        #class_variables = self.data_handler.get_class_variables()
        #class_variables += self.cloud_trainer.get_class_variables()
        super().__init__( **kwargs)




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
        
        # try:
        #     os.mkdir(folder)
        # except Exception:
        #     print("Could not create classifier project at given location")
        #     return 0

        # default_settings = os.path.join(self.default_path, self.settings_folder)
        # project_settings = os.path.join(self.project_path, self.settings_folder)

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
            Path to the directory where the clasifier is stored.
        """  
        self.set_project_path(path)
        self.data_handler.load_all(self.project_path)
        self.cloud_trainer.load_all(self.project_path)

    def save_project(self):
        self.data_handler.save_all(self.project_path)
        self.cloud_trainer.save_all(self.project_path)



    def set_parameters(self, **kwargs):
        self.data_handler.set_parameters(**kwargs)
        self.cloud_trainer.set_parameters(**kwargs)
        super().set_parameters(**kwargs)

    
    def set_project_path(self, path):
        self.project_path = path


