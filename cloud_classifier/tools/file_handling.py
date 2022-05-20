import random
import numpy as np
import os
import re


def create_subfolders(path, contains_filename=True):
    if contains_filename:
        path = os.path.split(path)[0]  # remove file nominator
    folders = path.split(os.sep)  # split path along os specific seperators
    current_path = ""
    for fol in folders:
        if not current_path:
            current_path = fol
        else:
            current_path = os.path.join(current_path, fol)
        if not os.path.exists(current_path):
            if current_path:
                os.mkdir(current_path)


def get_filename_pattern(structure, t_length):

    if "TIMESTAMP" not in structure:
        raise ValueError(
            "Specified file structure must contain region marked with string 'TIMESTAMP'"
        )

    replacemnt = "(.{" + str(t_length) + "})"
    pattern = structure.replace("TIMESTAMP", replacemnt)
    return re.compile(pattern)


def get_timestamp(reference_file, structure, ts_length):
    # get_filename if whole path is given
    name_split = os.path.split(reference_file)
    reference_file = name_split[1]

    # compute regex patterns
    pattern = get_filename_pattern(structure, ts_length)

    timestamp = pattern.match(reference_file)
    if not timestamp:
        raise Exception("Refernce data file does not match specified naming pattern")
    return timestamp.group(1)


def generate_filelist_from_folder(
    folder, satFile_pattern, labFile_pattern, only_satData=False
):
    """
    Extracts trainig files from folder
    Reads all matching files of satellite and label data from folder and adds them to project

    Parameters
    ----------
    folder : string
        Path to the folder containig the data files. +
        Default is True. If True, files will be read additive, if False old filelists will be overwritten.
    only_satData : bool
        Default is False. If False, filelist will contain sat data and labels, if True only sat_data files.
    """

    if folder is None:
        print("No folder specified!")
        return

    sat_files, lab_files = {}, {}
    files = os.listdir(folder)

    if only_satData:
        # return onky satelite date
        for file in files:
            sat_id = satFile_pattern.match(file)

            if sat_id:
                sat_files[sat_id.group(1)] = os.path.join(folder, file)
        return list(sat_files.values())
    else:
        for file in files:
            sat_id = satFile_pattern.match(file)
            lab_id = labFile_pattern.match(file)

            if sat_id:
                sat_files[sat_id.group(1)] = os.path.join(folder, file)
            elif lab_id:
                lab_files[lab_id.group(1)] = os.path.join(folder, file)

        # find pairs of sat data and labels
        training_sets = []
        for key in sat_files.keys():
            if key in lab_files:
                training_sets.append([sat_files[key], lab_files[key]])
        return training_sets


def split_sets(dataset, satFile_pattern, eval_size, timesensitive=True):
    """
    splits a set of data into an training and an evaluation set
    """
    eval_indeces = []
    timestamps = []
    if timesensitive:
        if eval_size % 24 != 0:
            raise ValueError(
                "When using timesensitive splitting eval_size must be multiple of 24"
            )
        timesorted = [[] for _ in range(24)]
        for i in range(len(dataset)):
            sat_id = satFile_pattern.match(os.path.basename(dataset[i][0]))
            if sat_id:
                timestamp = sat_id.group(1)
                timestamps.append(timestamp)
                hour = int(timestamp[-4:-2])
                timesorted[hour].append(i)
        n = int(eval_size / 24)
        for h in range(24):
            eval_indeces += random.sample(timesorted[h], n)
    else:
        eval_indeces = random.sample(range(len(dataset)), eval_size)

    training_indeces = [i for i in range(len(dataset)) if i not in eval_indeces]
    eval_set = [dataset[i] for i in eval_indeces]
    training_set = [dataset[i] for i in training_indeces]
    timestamps = [timestamps[i] for i in eval_indeces]

    return training_set, eval_set, timestamps


def get_label_name(sat_file, sat_file_structure, lab_file_structure, timestamp_length):
    timestamp = get_timestamp(sat_file, sat_file_structure, timestamp_length)
    label_file = lab_file_structure.replace("TIMESTAMP", timestamp)
    name, ext = os.path.splitext(label_file)
    label_file = name + "_predicted" + ext
    return label_file


def clean_eval_data(data_1, data_2):
    """
    returns one dimensional arrays without nan values

    Parameters
    ----------
    data_1, data_2 : array like
    """
    d1 = np.array(data_1).flatten()
    d2 = np.array(data_2).flatten()
    valid_1 = ~np.isnan(d1)
    valid_2 = ~np.isnan(d2)
    valid = np.logical_and(valid_1, valid_2)
    return d1[valid].astype(int), d2[valid].astype(int)


def get_match_string(labels, truth):

    correct = np.sum(labels == truth)
    not_nan = np.where(~np.isnan(labels))
    total = len(not_nan[0].flatten())

    return "Correctly identified {:.2f} %".format(100 * correct / total)
