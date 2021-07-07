import json
import os

class base_class:
    """
    Provides basic functionaltiry of parameter management and persistence 
    """


    def __init__(self, class_variables, **kwargs):

        dirname = os.path.dirname(__file__)
        self.default_path = os.path.join(dirname, "default")
        self.parameter_file = "parameters.json"
        self.data_file = "data.json"

        self.class_variables = class_variables
        # init all class parameterst
        self.__dict__.update((k,None) for k in class_variables)
        self.load_default_parameters()
        self.set_parameters(**kwargs)



    def set_parameters(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.class_variables)

    def get_parameters(self):
        return {k:self.__dict__[k] for k in self.class_variables}


    def get_class_variables(self):
        return self.class_variables


    def set_class_variables(self, class_variables):
        self.class_variables = class_variables



    def load_default_parameters(self):
        """
        Reads default json file for default parameters
        """
        self.load_parameters(os.path.join(self.default_path, self.parameter_file))



    def load_parameters(self, project_path):
        """
        Reads json file for parameters
        """
        path = os.path.join(project_path, self.parameter_file)
        with open(path, 'r') as parameters:
             kwargs = json.load(parameters)
             self.set_parameters(**kwargs)


    def save_parameters(self, project_path):
        """
        Save parameters into json file
        """
        path = os.path.join(project_path, self.parameter_file)
        parameters = {k:self.__dict__[k] for k in self.class_variables}
        json_obj = json.dumps(parameters, indent = 4)
        with open(path, "w") as outfile:
            outfile.write(json_obj)



        
