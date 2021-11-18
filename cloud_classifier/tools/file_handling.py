import xarray as xr
import numpy as np
import os
import re




def create_subfolders(path, project_path):
    path = os.path.normpath(path)
    folders = path.split(os.sep)
    current_path = project_path
    for fol in folders:
        current_path = os.path.join(current_path, fol)
        if(not os.path.isdir(current_path)):
            os.mkdir(current_path)

# def get_hour(filename, pattern):
#     timestamp = pattern.match(file).group(1)
#     return int(timestamp[-4:-2])


def get_filename_pattern(structure, t_length):

    if ("TIMESTAMP" not in structure):
        raise ValueError("Specified file structure must contain region marked as 'TIMESTAMP'")

    replacemnt =  "(.{" + str(t_length) + "})"
    pattern =  structure.replace("TIMESTAMP", replacemnt)
    return re.compile(pattern)



def get_timestamp(reference_file, structure, ts_length):
    # get_filename if whole path is given
    name_split = os.path.split(reference_file)
    reference_file = name_split[1]
    
    # compute regex patterns
    pattern = get_filename_pattern(structure, ts_length)

    timestamp = pattern.match(reference_file)
    if(not timestamp):
        raise Exception("Refernce data file does not match specified naming pattern")
    return timestamp.group(1)