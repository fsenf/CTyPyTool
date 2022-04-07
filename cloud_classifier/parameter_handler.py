import json
import os


class parameter_handler:
    """
    Provides functionaltiy of parameter management for cloud-classifier projects.
    Parameteres and filelists are provided as dictionaries, which are
    initalized with default values.
    Parameters and filelist can be loaded and saved in the file structure of
    a cloud classifier project
    """


    def __init__(self, path=None):
        """
        Declares position of parameter files and initalizes dictionaries.

        Parameters
        ----------
        path : path of project, optional
            Project path from which parameters and filelists are initalized.
            If None initialization will read from the 'defaults' folder.
        """
        self.__setting_files = [
            "config.json",
            "data_structure.json",
        ]

        self.__filelists_files = [
            "input_files.json",
            "evaluation_sets.json",
            "training_sets.json",
            "label_files.json",
        ]


        self.parameters = {
            "classifier_type": "Forest",
            "max_depth": 35,
            "ccp_alpha": 0,
            "n_estimators": 100,
            "feature_preselection": False,
            "max_features": None,
            "min_samples_split": 2,
            "merge_list": [],
            "difference_vectors": True,
            "original_values": True,
            "samples": 100,
            "data_source_folder": "../data/full_dataset",
            "timestamp_length": 13,
            "sat_file_structure": "msevi-medi-TIMESTAMP.nc",
            "label_file_structure": "nwcsaf_msevi-medi-TIMESTAMP.nc",
            "input_source_folder": "../data/example_data",
            "georef_file": "../data/auxilary_files/msevi-medi-georef.nc",
            "mask_file": "../data/auxilary_files/lsm_mask_medi.nc",
            "mask_key": "land_sea_mask",
            "mask_sea_coding": 0,
            "input_channels": [
                "bt062",
                "bt073",
                "bt087",
                "bt097",
                "bt108",
                "bt120",
                "bt134"
            ],
            "cloudtype_channel": "ct",
            "nwcsaf_in_version": "auto",
            "nwcsaf_out_version": "v2018",
            "hours": [
                0
            ]
        }

        self.filelists = {
            "training_sets": [],
            "label_files": [],
            "input_files": [],
            "evaluation_sets": [],
            "eval_timestamps": []
        }

        dirname = os.path.dirname(__file__)
        self.__default_path = os.path.join(dirname, "defaults")

        if (path is None):
            path = self.__default_path
        self.read_settings(path=path)


    def set_parameters(self, **kwargs):
        self.parameters.update((k, v) for k, v in kwargs.items()
                               if k in self.parameters)

    def set_filelists(self, **kwargs):
        self.filelists.update((k, v) for k, v in kwargs.items()
                              if k in self.filelists)

    def extend_filelists(self, **kwargs):
        for k in self.filelists:
            if (k in kwargs):
                self.filelists[k].extend(kwargs[k])

    def load_parameters(self, path):
        for file in self.__setting_files:
            filepath = os.path.join(path, "settings", file)
            self.__load_data(filepath=filepath, dictionary=self.parameters)

    def load_filelists(self, path):
        for file in self.__filelists_files:
            filepath = os.path.join(path, "filelists", file)
            self.__load_data(filepath=filepath, dictionary=self.filelists)

    def save_parameters(self, path):
        for file in self.__setting_files:
            filepath = os.path.join(path, "settings", file)
            self.__save_data(dictionary=self.parameters, filepath=filepath)

    def save_filelists(self, path):
        for file in self.__filelists_files:
            filepath = os.path.join(path, "filelists", file)
            self.__save_data(dictionary=self.filelists, filepath=filepath)



    def __load_data(self, filepath, dictionary):
        with open(filepath, 'r') as parameters:
            kwargs = json.load(parameters)
            dictionary.update((k, v) for k, v in kwargs.items()
                              if k in dictionary.keys())

    def __save_data(self, dictionary, filepath):
        # read saved data
        with open(filepath, 'r') as outfile:
            saved_data = json.load(outfile)
            # update saved data
            saved_data.update({k: dictionary[k] for k in saved_data.keys()
                               if k in dictionary})
            json_obj = json.dumps(saved_data, indent=4)
        # write json
        with open(filepath, 'w') as outfile:
            outfile.write(json_obj)
