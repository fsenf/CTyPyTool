import os

from . import parameter_handler


class cloud_project:

    """
    Basic architecture of a cloud classifier project.
    Base class that provides functionality of project creation, saving and loading
    as well as the setting of parameters via a parameter_handler.

    Attributes
    ----------
    filelists : dictionary
        Dictionary of filelists from the parameter_handler
    masked_indices : numpy array
        Indices of an applied mask file.
    param_handler : parameter_handler
        Parameter handler used to load, save and update project parameters and filelists
    params : dictionary
        Dictionary of parameters from the parameter_handler
    project_path : string
        Filepath of the cloud classifier project
    """

    def __init__(self, project_path=None):
        self.project_path = project_path
        self.param_handler = parameter_handler.parameter_handler()
        self.params = self.param_handler.parameters
        self.filelists = self.param_handler.filelists
        self.masked_indices = None

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
        """
        Loads project data from project directory.
        """
        if self.project_path is None:
            raise ValueError("Project path not set")
        self.param_handler.load_parameters(self.project_path)
        self.param_handler.load_filelists(self.project_path)

    def save_project_data(self):
        """
        Saves project data to project directory
        """
        if self.project_path is None:
            raise ValueError("Project path not set")
        self.param_handler.save_parameters(self.project_path)
        self.param_handler.save_filelists(self.project_path)

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

        if path is None:
            path = os.getcwd()

        folder = os.path.join(path, name)
        if os.path.isdir(folder):
            print("Folder with given name already exits! Loading existing project!")
        else:
            try:
                self.param_handler.initalize_settings(folder)
                print("Project folder created successfully!")

            except Exception:
                raise RuntimeError(
                    "Could not initalize project settings at given location!"
                )
        self.load_project(folder)

    def set_project_parameters(self, **kwargs):
        """
        Sets project parameters and saves them in the project files.

        Parameters
        ----------
        **kwargs
            Keywoard arguments. If the named argument is part of the project paramters or filelists
            the specified argument will be updated.
        """
        self.param_handler.set_parameters(**kwargs)
        self.param_handler.set_filelists(**kwargs)
        self.save_project_data()
