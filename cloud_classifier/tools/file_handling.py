import xarray as xr
import numpy as np
import os

# def get_filename_pattern(structure, t_length):
#     if ("TIMESTAMP" not in structure):
#         raise ValueError("Specified file structure must contain region marked as 'TIMESTAMP'")

#     replacemnt =  "(.{" + str(t_length) + "})"
#     pattern =  structure.replace("TIMESTAMP", replacemnt)
#     return re.compile(pattern)



def create_subfolder(path, project_path):
    path = os.path.normpath(path)
    folders = path.split(os.sep)
    current_path = project_path
    for fol in folders:
        current_path = os.path.join(current_path, fol)
        if(not os.path.isdir(current_path)):
            os.mkdir(current_path)

