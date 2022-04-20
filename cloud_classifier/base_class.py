import json
import os


class base_class:
    """
    Provides basic functionaltiry of parameter management and persistence
    """


    def __init__(self, **kwargs):

        self.setting_files = [
            "config.json",
            "data_structure.json",
        ]
        self.filelists = [
            "input_files.json",
            "evaluation_sets.json",
            "training_sets.json",
            "label_files.json",
        ]

        dirname = os.path.dirname(__file__)

        self.default_path = os.path.join(dirname, "defaults")


        if(not hasattr(self, 'class_variables')):
            print("Could not initalize class variables")
            return

        # init all class parameterst
        self.__dict__.update((k, None) for k in self.class_variables)
        self.load_data(self.default_path)
        # update with given parameters
        self.set_parameters(**kwargs)


    def init_class_variables(self, class_variables):
        if (not hasattr(self, 'class_variables')):
            self.class_variables = []
        self.class_variables = set(self.class_variables).union(set(class_variables))


    def load_data(self, path):
        for file in self.setting_files:
            filepath = os.path.join(path, "settings", file)
            self.load_parameters(filepath)
        for file in self.filelists:
            filepath = os.path.join(path, "filelists", file)
            self.load_parameters(filepath)

    def save_data(self, path):
        for file in self.setting_files:
            filepath = os.path.join(path, "settings", file)
            self.save_parameters(filepath)
        for file in self.filelists:
            filepath = os.path.join(path, "filelists", file)
            self.save_parameters(filepath)



    def set_parameters(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.class_variables)


    def load_parameters(self, filepath):
        """
        Reads json file for parameters
        """

        with open(filepath, 'r') as parameters:
            kwargs = json.load(parameters)
            self.set_parameters(**kwargs)


    def save_parameters(self, filepath):
        """
        Writes parameters
        """
        # get saved entries
        with open(filepath, 'r') as outfile:
            saved_params = json.load(outfile)
            # update saved params
            saved_params.update({k: self.__dict__[k] for k in saved_params.keys() if k in self.class_variables})
            json_obj = json.dumps(saved_params, indent = 4)

        # write json
        with open(filepath, 'w') as outfile:
            outfile.write(json_obj)
