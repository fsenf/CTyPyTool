import json
import os
import numpy as np
import shutil
import re
import pathlib

import cloud_trainer as ct
import data_handler as dh
import base_class as bc

import tools.file_handling as fh
import tools.confusion as conf


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
            print("Folder with given name already exits!")
        else:
            try:
                shutil.copytree(self.default_path, folder)
                print("Project folder created succefully!")

            except Exception:
                print("Could not initalize project settings at given location!")
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

    def run_training_pipeline(self, verbose = True, create_filelist = True, evaluation = False, create_training_data = True):
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






######################    Evaluation  ######################################
############################################################################


    def create_evaluation_plots(self, correlation = False, probas = False, comparison = False, 
        cmp_targets = None, plot_titles = None, show = True, verbose = True):

        for i in range(len(self.label_files)):
            label_file = self.label_files[i]
            truth_file = self.evaluation_sets[i][1]
            timestamp = self.eval_timestamps[i]
            if(correlation):
                self.save_coorMatrix(label_file = label_file, truth_file = truth_file, timestamp = timestamp, 
                    verbose=verbose, show = show)
            if(comparison):
                self.save_comparePlot(label_file = label_file, truth_file = truth_file, timestamp = timestamp, 
                    compare_projects =cmp_targets, plot_titles=plot_titles,verbose=verbose, show = show)
            if(probas):
                self.save_probasPlot(label_file = label_file, truth_file = truth_file, timestamp = timestamp, 
                    plot_titles=plot_titles, verbose=verbose, show = show)


    def get_overallCoocurrence(self, show = False):
        all_labels, all_truth = [], []
        for i in range(len(self.label_files)):
            label_file = self.label_files[i]
            truth_file = self.evaluation_sets[i][1]
            all_labels.append(self.get_plotable_data(data_file = label_file, get_coords = False))
            all_truth.append(self.get_plotable_data(data_file = truth_file, get_coords = False))
        all_labels, all_truth = fh.clean_eval_data(all_labels, all_truth)

        self.save_coorMatrix( label_data = all_labels, truth_data = all_truth, filename = "Overall_CoocurrenceMatrix.png", show = show)





    def save_comparePlot(self, label_file, truth_file, timestamp, compare_projects= None,
            plot_titles = None, verbose = True, show = True):

        all_files = [label_file]
        filename = os.path.split(label_file)[1]
        for proj_path in compare_projects:
            path = os.path.join(proj_path, "labels", filename)
            all_files.append(path)

        filename = timestamp + "_ComparisonPlot.png"
        path = os.path.join(self.project_path, "plots", "Comparisons", filename)
        fh.create_subfolders(path, self.project_path)

        hour = int(timestamp[-4:-2])
        self.plot_multiple(all_files, truth_file, georef_file = self.georef_file, reduce_to_mask = True,
            plot_titles = plot_titles, hour = hour, save_file = path, show = show)
        if (verbose):
            print("Comparison Plot saved at " + path)


    def save_probasPlot(self, label_file, truth_file, timestamp,
            plot_titles = None, verbose = True, show = True, filename = None):


        if (filename is None):
            filename = timestamp + "_ProbabilityPlot.png"
        path = os.path.join(self.project_path, "plots", "Probabilities", filename)
        fh.create_subfolders(path, self.project_path)

        hour = int(timestamp[-4:-2])
        self.plot_probas(label_file, truth_file, georef_file = self.georef_file, reduce_to_mask = True,
            plot_titles = plot_titles, hour = hour, save_file = path, show = show)
        if (verbose):
            print("Probability Plot saved at " + path)


    def save_coorMatrix(self, label_file = None, truth_file = None,
            label_data = None, truth_data = None,
            timestamp = None, filename = None,
            normalize = True, verbose = True, show = True):

        if (truth_file is None and truth_data is None):
            raise ValueError("'truth_file' or 'truth_data' be specified!")
        if (label_file is None and label_data is None):
            raise ValueError("'label_file' or 'label_data' be specified!")
        if (filename is None and timestamp is None):
            raise ValueError("'filename' or 'timestamp' be specified!")

        if (filename is None):
            filename = timestamp + "_CoocurrenceMatrix.png"
        if(label_data is None):
            label_data = self.get_plotable_data(data_file = label_file, reduce_to_mask = True, get_coords = False)
        if(truth_data is None):
            truth_data = self.get_plotable_data(data_file = truth_file, reduce_to_mask = True, get_coords = False)

        path = os.path.join(self.project_path, "plots", "Coocurrence", filename)
        fh.create_subfolders(path)

        conf.plot_coocurrence_matrix(label_data, truth_data, normalize=normalize, save_file = path)
        if (verbose):
            print("Correlation Matrix saved at " + path, filename)








    #############           Steps of the pipeline         ######################
    ##########################################################################


    #### training
    def create_training_filelist(self, verbose = True):
        satFile_pattern = fh.get_filename_pattern(self.sat_file_structure, self.timestamp_length)
        labFile_pattern = fh.get_filename_pattern(self.label_file_structure, self.timestamp_length)
        self.training_sets =  fh.generate_filelist_from_folder(self.data_source_folder, satFile_pattern, labFile_pattern)
        filepath = os.path.join(self.project_path, "filelists", "training_sets.json")
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

        if pathlib.Path(ref_path).is_file():
            self.reference_file = ref_path
            if (verbose):
                print("Reference file found")
        else:
            self.create_reference_file()

    ### predicting
    def extract_input_filelist(self, verbose = True):
        satFile_pattern = fh.get_filename_pattern(self.sat_file_structure, self.timestamp_length)
        labFile_pattern = fh.get_filename_pattern(self.label_file_structure, self.timestamp_length)

        self.input_files =  fh.generate_filelist_from_folder(folder = self.input_source_folder,
            satFile_pattern = satFile_pattern,
            labFile_pattern = labFile_pattern,
            only_sataData = True)
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
        vectors, indices = super().create_input_vectors(file)
        if(verbose):
                print("Input vectors created!")
        return vectors, indices


    def predict_labels(self, input_vectors, verbose = True):
        labels = super().predict_labels(input_vectors)
        if(verbose):
            print("Predicted Labels!")
        return labels

    def save_labels(self, labels, indices, sat_file, probas = None, verbose = True):
        name = fh.get_label_name(sat_file, self.sat_file_structure, self.label_file_structure, self.timestamp_length)
        filepath = os.path.join(self.project_path, "labels", name)
        fh.create_subfolders(filepath)
        self.make_xrData(labels, indices, NETCDF_out = filepath, prob_data = probas)
        if(verbose):
            print("Labels saved as " + name )
        return filepath


    ### evaluation
    def create_split_training_filelist(self):
        satFile_pattern = fh.get_filename_pattern(self.sat_file_structure, self.timestamp_length)
        labFile_pattern = fh.get_filename_pattern(self.label_file_structure, self.timestamp_length)
        datasets =  fh.generate_filelist_from_folder(self.data_source_folder, satFile_pattern, labFile_pattern)

        self.training_sets, self.evaluation_sets, self.timestamps = fh.split_sets(datasets, satFile_pattern, 24, timesensitive = True)
        self.input_files = [s[0] for s in self.evaluation_sets]
        self.save_project_data()
