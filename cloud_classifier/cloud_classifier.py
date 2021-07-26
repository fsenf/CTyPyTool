import json
import os
import numpy as np
import re

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
        class_variables =   ["data_source_folder", 
                            "timestamp_length",
                            "sat_file_structure",
                            "label_file_structure"]
        class_variables += self.data_handler.get_class_variables()
        class_variables += self.cloud_trainer.get_class_variables()
        super().__init__(class_variables, **kwargs)




    def create_new_project(self, name, path = None):
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
        
        super().create_new_project(folder)
        self.save_parameters(type = "config")
        self.save_parameters(type = "data_structure")
        self.save_parameters(type = "training_data")




    def generate_filelist_from_folder(self, folder = None, additive = True):
        """
        Extracts trainig files from folder
        Reads all matching files of satellite and label data from folder and adds them to project

        Parameters
        ----------
        folder : string (Optional)
            Path to the folder containig the data files. If none is given path will be read from settings
        additive : bool
            If True, files will be read additive, if False old filelists will be overwritten.
        """  
        


        if (folder is None):
            folder = self.data_source_folder
        if (folder is None):
            print("No folder specified!")
            return


        if ("TIMESTAMP" not in self.sat_file_structure or 
            "TIMESTAMP" not in self.label_file_structure):
            print ("Specified file name must contain region marked as 'TIMESTAMP'")
            return


        pattern = "(.{" + str(self.timestamp_length) + "})"

        sat_pattern = self.sat_file_structure.replace("TIMESTAMP", pattern)
        lab_pattern = self.label_file_structure.replace("TIMESTAMP", pattern)
        # sat_pattern = "msevi-medi-(.{13})\.nc"
        # lab_pattern = "nwcsaf_msevi-medi-(.{13})\.nc"

        sat_comp = re.compile(sat_pattern)
        lab_comp = re.compile(lab_pattern)
        data_sets, sat_files, lab_files = list(), {}, {}

        files = os.listdir(folder)
        for file in files:
            sat_id = sat_comp.match(file)
            lab_id = lab_comp.match(file)
            
            if (sat_id):
                sat_files[sat_id.group(1)] = folder  + file  
            elif (lab_id):
                lab_files[lab_id.group(1)] = folder  + file

        for key in sat_files.keys():
            if(key in lab_files):
                data_sets.append([sat_files[key], lab_files[key]])
        

        print(data_sets)

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



