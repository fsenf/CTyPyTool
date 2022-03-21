
import json
import os


class parameter_handler:
    """
    Provides functionaltiy of parameter management for cloud-classifier projects.
    """


    def __init__(self, path = None):

        self.__setting_files = [
            "config.json",
            "data_structure.json",
        ]
        self.__filelists = [
            "input_files.json",
            "evaluation_sets.json",
            "training_sets.json",
            "label_files.json",
        ]

        dirname = os.path.dirname(__file__)
        self.__default_path = os.path.join(dirname, "defaults")

        # init all class parameterst
        self.__dict__.update((k, None) for k in self.__parameters)
        self.load_data(self.__default_path)
        # update with given parameters

    def read_settings(self, path):
        """
        Reads project settings from of project folder at 'path'.

        Parameters
        ----------
        path : string
            Path of the project,
        """
        