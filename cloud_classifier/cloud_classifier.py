import json
import os
import numpy as np
import shutil
import re
import random
from pathlib import Path

import cloud_trainer as ct
import data_handler as dh
import base_class as bc
import tools.file_handling as fh

import importlib
importlib.reload(ct)
importlib.reload(dh)
importlib.reload(bc)
importlib.reload(fh)

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
            "input_files",
            "evaluation_sets",
            "label_files",
            "eval_timestamps"
            }
        self.project_path = None

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
        if (os.path.isdir(folder)):
            print("Folder with given name already exits")
        else:
            try:
                shutil.copytree(self.default_path, folder)

            except Exception:
                print("Could not initalize project settings at given location")
                return 0
        self.load_project(folder)


    def load_project(self, path):
        """
        Loads a persistant classifier project.

        Parameters
        ----------
        path : string 
            Path to the stored project
        """  
        self.project_path = path
        self.load_project_data()


    # def set_project_path(self, path):
    #     self.project_path = path


    def load_project_data(self, path = None):
        if (path is None):
            path = self.project_path 
        if (path is None):
            raise ValueError("Project path not set")
        self.load_data(path)

    def save_project_data(self, path = None):
        if (path is None):
            path = self.project_path 
        if (path is None):
            raise ValueError("Project path not set")
        self.save_data(path)


    def set_project_parameters(self, **kwargs):
        self.set_parameters(**kwargs)
        if(not self.project_path is None):
            self.save_data(self.project_path)





    ######################    PIPELINE  ######################################
    ##########################################################################

    def run_training_pipeline(self, verbose = True, create_filelist = True, evaluation = False, create_training_data = False):
        if (create_filelist):
            if (evaluation):
                self.create_split_training_filelist()
            else:
                self.create_training_filelist(verbose = verbose)
        if (self.reference_file is None):
            self.create_reference_file()
        self.apply_mask(verbose = verbose)
        if(create_training_data):
            v,l = self.create_training_set(verbose = verbose)
        else: 
            v,l = self.load_training_set()
        self.train_classifier(v,l, verbose = verbose)
        self.save_project_data()


    def run_prediction_pipeline(self, verbose = True, create_filelist = True, evaluation = False):

        if (create_filelist and not evaluation):
            self.extract_input_filelist(verbose = verbose)

        self.load_classifier(reload = True, verbose = verbose)
        self.apply_mask(verbose = verbose)
        self.set_reference_file(verbose = verbose)
        self.label_files = []
        for file in self.input_files:
            vectors, indices = self.create_input_vectors(file, verbose = verbose)
            probas = None
            if(self.classifier_type == "Forest"):
                li = self.classifier.classes_
                probas = self.get_forest_proabilties(vectors)
                labels = [li[i] for i in np.argmax(probas, axis = 1)]
            else:
                labels = self.predict_labels(vectors, verbose = verbose)

            filename = self.save_labels(labels, indices, file, probas, verbose = verbose)
            self.label_files.append(filename)


        self.save_project_data()
            #TODO: convert and save labels

    def evaluation_plots(self, verbose=True, correlation = False, probas = False, 
        cross = False, cross_partner = None):

        for i in range(len(self.label_files)):
            if(correlation):
                self.save_evaluation_coorMatrix(i, verbose=verbose)



    def save_evaluation_coorMatrix(self, index, normalize = True, verbose=True):
        label_file = self.label_files[index]
        truth_file = self.evaluation_sets[index][1]

        path = os.path.join("plots", "Coocurrence")
        fh.create_subfolder(path, self.project_path)

        ts = self.get_timestamp(truth_file, self.label_file_structure, self.timestamp_length)
        filename = ts + "_correlation_Matrix.png"
        path = os.path.join(self.project_path, path, filename)

        self.plot_coocurrence_matrix(label_file, truth_file, normalize, path)
        if (verbose):
            print("Correlation Matrix saved at " + path)


    #############           Steps of the pipeline         ######################
    ##########################################################################


    #### training
    def create_training_filelist(self, verbose = True):
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

    def load_training_set(self, verbose = True):
        filename = os.path.join(self.project_path, "data", "training_data")
        v,l = super().load_training_set(filename)
        if (verbose):
            print("Training data loaded!")
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
        self.save_project_data()

    def set_reference_file(self, verbose = True):
        ref_path = os.path.join(self.project_path, "data", "label_reference.nc")

        if Path(ref_path).is_file():
            self.reference_file = ref_path
            if (verbose):
                print("Reference file found")
        else:
            self.create_reference_file()

    ### predicting
    def extract_input_filelist(self, verbose = True):
        self.input_files =  self.generate_filelist_from_folder(folder = self.input_source_folder, no_labels = True)
        filepath = os.path.join(self.project_path, "filelists", "input_files.json")
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

    def save_labels(self, labels, indices, sat_file, probas = None, verbose = True):
        name = self.get_label_name(sat_file)
        filepath = os.path.join(self.project_path, "labels", name)
        self.make_xrData(labels, indices, NETCDF_out = filepath, prob_data = probas)
        if(verbose):
            print("Labels saved as " + name )
        return filepath


    ### evaluation
    def create_split_training_filelist(self):
        datasets =  self.generate_filelist_from_folder(self.data_source_folder)
        self.training_sets, self.evaluation_sets, self.timestamps = self.split_sets(datasets, 24, timesensitive = True)
        self.input_files = [s[0] for s in self.evaluation_sets]
        self.save_project_data()



    ##########################################################################
    #### TODO: externalize


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





    def split_sets(self, dataset, eval_size = 24, timesensitive = True):
        """
        splits a set of data into an training and an evaluation set
        """
        sat_pattern = self.get_filename_pattern(self.sat_file_structure, self.timestamp_length)

        eval_indeces = []
        timestamps = []
        if(timesensitive):
            if(eval_size % 24 != 0):
                raise ValueError("When using timesensitive splitting eval_size must be multiple of 24")
            timesorted = [[] for _ in range(24)]
            for i in range(len(dataset)):
                sat_id = sat_pattern.match(os.path.basename(dataset[i][0]))
                if(sat_id):
                    timestamp = sat_id.group(1)
                    timestamps.append(timestamp)
                    hour = int(timestamp[-4:-2])
                    timesorted[hour].append(i)
            n = int(eval_size/24)
            for h in range(24):
                eval_indeces += random.sample(timesorted[h], n)
        else:
            eval_indeces = random.sample(range(len(dataset)), eval_size)

        training_indeces = [i for i in range(len(dataset)) if i not in eval_indeces]
        eval_set = [dataset[i] for i in eval_indeces]
        training_set = [dataset[i] for i in training_indeces]

        return training_set, eval_set, timestamps





    def get_hour(self, filename, pattern):
        timestamp = pattern.match(file).group(1)
        return int(timestamp[-4:-2])


    def get_label_name(self, sat_file):
        timestamp = get_timestamp(sat_file, self.sat_file_structure, self.timestamp_length)
        label_file = self.label_file_structure.replace("TIMESTAMP", timestamp)
        name, ext = os.path.splitext(label_file)
        label_file = name + "_predicted" + ext
        return label_file


    def get_timestamp(self, reference_file, structure, ts_length):
        # get_filename if whole path is given
        name_split = os.path.split(reference_file)
        reference_file = name_split[1]
        
        # compute regex patterns
        pattern = self.get_filename_pattern(structure, ts_length)

        timestamp = pattern.match(reference_file)
        if(not timestamp):
            raise Exception("Refernce data file does not match specified naming pattern")
        return timestamp.group(1)


    def get_filename_pattern(self, structure, t_length):

        if ("TIMESTAMP" not in structure):
            raise ValueError("Specified file structure must contain region marked as 'TIMESTAMP'")

        replacemnt =  "(.{" + str(t_length) + "})"
        pattern =  structure.replace("TIMESTAMP", replacemnt)
        return re.compile(pattern)