
import os
import numpy as np

import parameter_handler
import cloud_trainer

import tools.data_handling as dh
import tools.file_handling as fh
import tools.confusion as conf
import tools.write_netcdf as ncdf


import importlib
importlib.reload(cloud_trainer)
importlib.reload(parameter_handler)

importlib.reload(dh)
importlib.reload(fh)
importlib.reload(ncdf)
importlib.reload(conf)




class cloud_classifier():


    def __init__(self):
        self.project_path = None
        self.param_handler = parameter_handler.parameter_handler()

        self.params = self.param_handler.parameters
        self.filelists = self.param_handler.filelists

        self.trainer = cloud_trainer.cloud_trainer()
        self.masked_indices = None


    # ############ CREATING, LOADING AND SAVING PROJECTS ######################
    # ########################################################################


    def create_new_project(self, name, path=None):
        """
        Creates a persistant classifier project.


        Parameters
        ----------
        name : string
            Name of the the project that will be created

        path : string, optional
            Path to the directory where the project will be stored. If none is
            given, the current working directory will be used.
        """

        if (path is None):
            path = os.getcwd()

        folder = os.path.join(path, name)
        if (os.path.isdir(folder)):
            print("Folder with given name already exits! Loading existing project!")
        else:
            try:
                self.param_handler.initalize_settings(folder)
                print("Project folder created successfully!")

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



    def load_project_data(self):
        if (self.project_path is None):
            raise ValueError("Project path not set")
        self.param_handler.load_parameters(self.project_path)
        self.param_handler.load_filelists(self.project_path)


    def save_project_data(self):
        if (self.project_path is None):
            raise ValueError("Project path not set")
        self.param_handler.save_parameters(self.project_path)
        self.param_handler.save_filelists(self.project_path)


    def set_project_parameters(self, **kwargs):
        self.param_handler.set_parameters(**kwargs)
        self.param_handler.set_filelists(**kwargs)
        self.save_project_data()




    ######################    PIPELINE  ######################################
    ##########################################################################



    def run_training_pipeline(self, verbose = True, create_filelist = True, evaluation = False,
                              create_training_data = True):
        if (create_filelist):
            if (evaluation):
                self.create_split_training_filelist()
            else:
                self.create_training_filelist(verbose = verbose)

        ncdf.create_reference_file(self.project_path, self.filelists["training_sets"],
                                   self.params["cloudtype_channel"])

        self.apply_mask(verbose = verbose)

        if(create_training_data):
            vectors, labels = self.create_training_data(verbose = verbose)
        else:
            vectors, labels = self.load_training_set()

        self.trainer.train_classifier(vectors, labels, self.params, verbose)
        self.trainer.save_classifier(self.project_path, verbose)


    def run_prediction_pipeline(self, verbose = True, create_filelist = True, evaluation = False):

        if (create_filelist and not evaluation):
            self.extract_input_filelist(verbose = verbose)

        self.trainer.load_classifier(self.project_path, verbose = verbose)
        self.apply_mask(verbose = verbose)

        ncdf.create_reference_file(self.project_path, self.filelists["training_sets"],
                                   self.params["cloudtype_channel"])

        label_files = []
        for file in self.filelists["input_files"]:
            vectors, indices = dh.create_input_vectors(filename = file, params = self.params,
                                                       indices = self.masked_indices, verbose = verbose)
            probas = None
            # when classifier is Forest, get vote share for each type
            if(self.params["classifier_type"] == "Forest"):
                li = self.classifier.classes_
                probas = self.trainer.get_forest_proabilties(vectors, self.params)
                labels = [li[i] for i in np.argmax(probas, axis = 1)]
            # else get only labels
            else:
                labels = self.trainer.predict_labels(vectors, self.params)

            filename = self.write_labels(labels, indices, file, probas, verbose = verbose)
            label_files.append(filename)
        self.param_handler.set_filelists(label_files = label_files)
        self.param_handler.save_filelists(self.project_path)


    #############           Steps of the pipeline         ######################
    ##########################################################################


    def create_training_filelist(self, verbose = True):

        satFile_pattern = fh.get_filename_pattern(self.params["sat_file_structure"],
                                                  self.params["timestamp_length"])
        labFile_pattern = fh.get_filename_pattern(self.params["label_file_structure"],
                                                  self.params["timestamp_length"])
        training_sets = fh.generate_filelist_from_folder(self.params["data_source_folder"],
                                                         satFile_pattern, labFile_pattern)

        self.param_handler.set_filelists(training_sets = training_sets)
        self.param_handler.save_filelists(self.project_path)
        if (verbose):
            print("Filelist created!")


    def apply_mask(self, verbose = True):
        self.masked_indices = dh.set_indices_from_mask(self.params)
        if (verbose):
            print("Masked indices set!")


    def create_training_data(self, verbose = True):
        vectors, labels = dh.create_training_vectors(self.params, self.filelists["training_sets"],
                                                     self.masked_indices, verbose = verbose)
        dh.save_training_data(vectors, labels, self.project_path)
        if (verbose):
            print("Training data created!")
        return vectors, labels


    def load_training_set(self, verbose = True):
        vectors, labels = dh.load_training_data(self.project_path)
        if (verbose):
            print("Training data loaded!")
        return vectors, labels


    def train_classifier(self, vectors, labels, verbose = True):
        self.trainer.train_classifier(vectors, labels, self.params)
        self.trainer.save_classifier(self.project_path)
        if (verbose):
            print("Classifier created!")




    def extract_input_filelist(self, verbose = True):
        satFile_pattern = fh.get_filename_pattern(self.params["sat_file_structure"],
                                                  self.params["timestamp_length"])
        labFile_pattern = fh.get_filename_pattern(self.params["label_file_structure"],
                                                  self.params["timestamp_length"])

        input_files = fh.generate_filelist_from_folder(folder = self.params["input_source_folder"],
                                                    satFile_pattern = satFile_pattern,
                                                    labFile_pattern = labFile_pattern,
                                                    only_sataData = True)

        self.param_handler.set_filelists(input_files = input_files)
        self.param_handler.save_filelists(self.project_path)

        if (verbose):
            print("Input filelist created!")



    def write_labels(self, labels, indices, sat_file, probas = None, verbose = True):
        name = fh.get_label_name(sat_file = sat_file,
                                 sat_file_structure = self.params["sat_file_structure"],
                                 lab_file_structure = self.params["label_file_structure"],
                                 timestamp_length = self.params["timestamp_length"])

        filepath = os.path.join(self.project_path, "labels", name)
        fh.create_subfolders(filepath)
        ncdf.make_xrData(labels = labels, indices = indices,
                         project_path = self.project_path,
                         ct_channel = self.params["cloudtype_channel"],
                         NETCDF_out = filepath, prob_data = probas)
        if(verbose):
            print("Labels saved as " + name)
        return filepath













    ### evaluation
    def create_split_training_filelist(self):
        satFile_pattern = fh.get_filename_pattern(self.params["sat_file_structure"],
                                                  self.params["timestamp_length"])
        labFile_pattern = fh.get_filename_pattern(self.params["label_file_structure"],
                                                  self.params["timestamp_length"])
        datasets =  fh.generate_filelist_from_folder(self.data_source_folder, satFile_pattern, labFile_pattern)

        self.training_sets, self.evaluation_sets, self.timestamps = fh.split_sets(datasets, satFile_pattern, 24, timesensitive = True)
        self.input_files = [s[0] for s in self.evaluation_sets]
        self.save_project_data()







######################    Evaluation  ######################################
############################################################################


    def create_evaluation_plots(self, correlation = False, probas = False, comparison = False,
                                cmp_targets = None, plot_titles = None, show = True, verbose = True):

        for i in range(len(self.label_files)):
            label_file = self.label_files[i]
            truth_file = self.evaluation_sets[i][1]
            timestamp = self.eval_timestamps[i]
            if(correlation):
                self.save_coorMatrix(label_file = label_file, truth_file = truth_file,
                                     timestamp = timestamp, verbose=verbose, show = show)
            if (comparison):
                self.save_comparePlot(label_file = label_file, truth_file = truth_file,
                                      timestamp = timestamp, compare_projects =cmp_targets, plot_titles=plot_titles, verbose=verbose, show = show)
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


