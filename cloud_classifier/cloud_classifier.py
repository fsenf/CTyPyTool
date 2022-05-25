import os
import numpy as np

import parameter_handler
import cloud_trainer

import tools.data_handling as dh
import tools.file_handling as fh
import tools.confusion as conf
import tools.write_netcdf as ncdf

import cloud_project

import importlib

importlib.reload(cloud_trainer)
importlib.reload(parameter_handler)

importlib.reload(dh)
importlib.reload(fh)
importlib.reload(ncdf)
importlib.reload(conf)


class cloud_classifier(cloud_project.cloud_project):

    """
    Cloud classifier class building on cloud_project class. Main class of the cloud classifier framework.
    Contains function that implement complete training and prediction procedures.

    """

    def __init__(self, project_path=None):

        super().__init__(project_path)
        self.__trainer = cloud_trainer.cloud_trainer()

    ######################    PIPELINE  ######################################
    ##########################################################################

    def run_training_pipeline(
        self,
        verbose=True,
        create_filelist=True,
        create_training_data=True,
    ):
        """
        Training pipeline that creates training data and trains a classifier according to
        the project parameters.

        Parameters
        ----------
        verbose : bool, optional
            If True, the function will give detailed command line output.
        create_filelist : bool, optional
            If True, a filelist will be created from the folder specified in the project settings.
            If False, the fielelist will be used as present in the project folder.
        create_training_data : bool, optional
            If True, training data will be created according to the project settings.
            If False, the training data will be used as present in the project folder.
        """
        self.load_project_data()
        if create_filelist:
            self.__create_training_filelist(verbose=verbose)

        ncdf.create_reference_file(
            self.project_path,
            self.filelists["training_sets"],
            self.params["cloudtype_channel"],
        )

        self.__apply_mask(verbose=verbose)

        if create_training_data:
            vectors, labels = self.__create_training_data(verbose=verbose)
        else:
            vectors, labels = self.__load_training_set()

        self.__trainer.train_classifier(vectors, labels, self.params, verbose)
        self.__trainer.save_classifier(self.project_path, verbose)

    def run_prediction_pipeline(self, verbose=True, create_filelist=True):
        """
        Prediction pipeline that creates predicted labels using a previously trained classifier according
        to the project parameters.

        Parameters
        ----------
        verbose : bool, optional
            If True, the function will give detailed command line output.
        create_filelist : bool, optional
            If True, a filelist will be created from the folder specified in the project settings.
            If False, the fielelist will be used as present in the project folder.
        """
        self.load_project_data()
        if create_filelist:
            self.__extract_input_filelist(verbose=verbose)

        self.__trainer.load_classifier(self.project_path, verbose=verbose)
        self.__apply_mask(verbose=verbose)

        ncdf.create_reference_file(
            self.project_path,
            self.filelists["training_sets"],
            self.params["cloudtype_channel"],
        )

        label_files = []
        for file in self.filelists["input_files"]:
            vectors, indices = dh.create_input_vectors(
                filename=file,
                params=self.params,
                indices=self.masked_indices,
                verbose=verbose,
            )
            probas = None
            # when classifier is Forest, get vote share for each type
            if self.params["classifier_type"] == "Forest":
                li = self.__trainer.classifier.classes_
                probas = self.__trainer.get_forest_proabilties(vectors, self.params)
                labels = [li[i] for i in np.argmax(probas, axis=1)]
            # else get only labels
            else:
                labels = self.__trainer.predict_labels(vectors, self.params)

            filename = self.__write_labels(
                labels, indices, file, probas, verbose=verbose
            )
            label_files.append(filename)
        self.param_handler.set_filelists(label_files=label_files)
        self.param_handler.save_filelists(self.project_path)

    #############           Steps of the pipeline         ######################
    ##########################################################################

    def __create_training_filelist(self, verbose=True):

        satFile_pattern = fh.get_filename_pattern(
            self.params["sat_file_structure"], self.params["timestamp_length"]
        )
        labFile_pattern = fh.get_filename_pattern(
            self.params["label_file_structure"], self.params["timestamp_length"]
        )
        training_sets = fh.generate_filelist_from_folder(
            self.params["data_source_folder"], satFile_pattern, labFile_pattern
        )

        self.param_handler.set_filelists(training_sets=training_sets)
        self.param_handler.save_filelists(self.project_path)
        if verbose:
            print("Filelist created!")

    def __apply_mask(self, verbose=True):
        self.masked_indices = dh.set_indices_from_mask(self.params)
        if verbose:
            print("Masked indices set!")

    def __create_training_data(self, verbose=True):
        vectors, labels = dh.create_training_vectors(
            self.params,
            self.filelists["training_sets"],
            self.masked_indices,
            verbose=verbose,
        )
        dh.save_training_data(vectors, labels, self.project_path)
        if verbose:
            print("Training data created!")
        return vectors, labels

    def __load_training_set(self, verbose=True):
        vectors, labels = dh.load_training_data(self.project_path)
        if verbose:
            print("Training data loaded!")
        return vectors, labels

    def __extract_input_filelist(self, verbose=True):
        satFile_pattern = fh.get_filename_pattern(
            self.params["sat_file_structure"], self.params["timestamp_length"]
        )
        labFile_pattern = fh.get_filename_pattern(
            self.params["label_file_structure"], self.params["timestamp_length"]
        )

        input_files = fh.generate_filelist_from_folder(
            folder=self.params["input_source_folder"],
            satFile_pattern=satFile_pattern,
            labFile_pattern=labFile_pattern,
            only_satData=True,
        )

        self.param_handler.set_filelists(input_files=input_files)
        self.param_handler.save_filelists(self.project_path)

        if verbose:
            print("Input filelist created!")

    def __write_labels(self, labels, indices, sat_file, probas=None, verbose=True):
        name = fh.get_label_name(
            sat_file=sat_file,
            sat_file_structure=self.params["sat_file_structure"],
            lab_file_structure=self.params["label_file_structure"],
            timestamp_length=self.params["timestamp_length"],
        )

        filepath = os.path.join(self.project_path, "labels", name)
        fh.create_subfolders(filepath)
        ncdf.make_xrData(
            labels=labels,
            indices=indices,
            project_path=self.project_path,
            ct_channel=self.params["cloudtype_channel"],
            NETCDF_out=filepath,
            prob_data=probas,
        )
        if verbose:
            print("Labels saved as " + name)
        return filepath
