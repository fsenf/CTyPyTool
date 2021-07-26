import json
import os
import shutil

class base_class:
    """
    Provides basic functionaltiry of parameter management and persistence 
    """


    def __init__(self, class_variables, project_path = None, **kwargs):

        dirname = os.path.dirname(__file__)
        self.default_path = os.path.join(dirname, "defaults")

        self.settings_folder = "settings"
        self.config_file = os.path.join(self.settings_folder,"config.json")
        self.data_file = os.path.join(self.settings_folder,"training_data.json")
        self.data_structure = os.path.join(self.settings_folder,"data_structure.json")

        self.project_path = project_path
        self.class_variables = class_variables

        # init all class parameterst
        self.__dict__.update((k,None) for k in class_variables)
        self.load_parameters(type = "config", default = True)
        self.load_parameters(type = "training_data", default = True)
        self.load_parameters(type = "data_structure", default = True)

        self.set_parameters(**kwargs)



    def set_parameters(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.class_variables)



    def set_project_path(self, path):
        self.project_path = path



    def create_new_project(self, folder):
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
        
        try:
            os.mkdir(folder)
        except Exception:
            print("Could not create classifier project at given location")
            return 0

        self.set_project_path(folder)
        default_settings = os.path.join(self.default_path, self.settings_folder)
        project_settings = os.path.join(self.project_path, self.settings_folder)

        try:
            shutil.copytree(default_settings, project_settings)
        except Exception:
            print("Could not initalize project settings at given location")
            return 0


    def load_parameters(self, type = "config", default = False):
        """
        """
        if (default):
            path = self.default_path
        else:
            path = self.project_path
        if (path is None):
            print("No project path set!")
            return

        file = self.__parse_parameter_type(type)
        kwargs = self.__read_parameters(os.path.join(path, file))
        self.set_parameters(**kwargs)
        return kwargs



    def save_parameters(self, type = "config"):
        """
        """
        if (self.project_path is None):
            print("No project path set!")
            return

        file = self.__parse_parameter_type(type)
        self.__write_parameters(os.path.join(self.project_path, file))



    def get_class_variables(self):
        return self.class_variables



    def __read_parameters(self, filepath):
        """
        Reads json file for parameters
        """
        with open(filepath, 'r') as parameters:
            kwargs =  json.load(parameters)
        return kwargs



    def __write_parameters(self, filepath, **kwargs):
        """
        Writes parameters
        """
        # get saved entries
        with open(filepath, 'r') as outfile:
            saved_params =  json.load(outfile)
            # update saved params
            saved_params.update({k:self.__dict__[k] for k in saved_params.keys()})
            json_obj = json.dumps(saved_params, indent = 4)

        # write json
        with open(filepath, 'w') as outfile:
             outfile.write(json_obj)



    def __parse_parameter_type(self, type):
        """
        Match type argument to settings file
        """
        file = None
        if (type == "config"):
            file = self.config_file
        elif (type == "training_data"):
            file = self.data_file
        elif (type == "data_structure"):
            file = self.data_structure
        else:
            print("No valid parameter type given")
        return file
