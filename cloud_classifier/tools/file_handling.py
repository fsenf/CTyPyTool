import random
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


def generate_filelist_from_folder(folder, satFile_pattern, labFile_pattern, only_sataData = False):
    """
    Extracts trainig files from folder
    Reads all matching files of satellite and label data from folder and adds them to project

    Parameters
    ----------
    folder : string 
        Path to the folder containig the data files. +
        Default is True. If True, files will be read additive, if False old filelists will be overwritten.
    only_sataData : bool 
        Default is False. If False, filelist will contain sat data and labels, if True only sat_data files.
    """

    if (folder is None):
        print("No folder specified!")
        return

    sat_files, lab_files = {}, {}
    files = os.listdir(folder)

    if (only_sataData):
        # return onky satelite date 
        for file in files:
            sat_id = satFile_pattern.match(file)
   
            if (sat_id):
                sat_files[sat_id.group(1)] = os.path.join(folder, file)
        return list(sat_files.values())
    else:
        for file in files:
            sat_id = satFile_pattern.match(file)
            lab_id = labFile_pattern.match(file)
            
            if (sat_id):
                sat_files[sat_id.group(1)] = os.path.join(folder, file)
            elif (lab_id):
                lab_files[lab_id.group(1)] = os.path.join(folder, file)

        # find pairs of sat data and labels
        training_sets = []
        for key in sat_files.keys():
            if(key in lab_files):
                training_sets.append([sat_files[key], lab_files[key]])
        return  training_sets


def split_sets(dataset, satFile_pattern, eval_size = 24, timesensitive = True):
    """
    splits a set of data into an training and an evaluation set
    """
    eval_indeces = []
    timestamps = []
    if(timesensitive):
        if(eval_size % 24 != 0):
            raise ValueError("When using timesensitive splitting eval_size must be multiple of 24")
        timesorted = [[] for _ in range(24)]
        for i in range(len(dataset)):
            sat_id = satFile_pattern.match(os.path.basename(dataset[i][0]))
            if(sat_id):
                timestamp = sat_id.group(1)
                timestamps.append(timestamp)
                hour = int(timestamp[-4:-2])
                timesorted[hour].append(i)
        n = int(eval_size/24)
        for h in range(24):
            eval_indeces += random.sample(timesorted[h], n)
    else:
        eval_indeces = random.sample(range(len(dataset)), eval_size)

    training_indeces = [i for i in range(len(dataset)) if i not in eval_indeces]
    eval_set = [dataset[i] for i in eval_indeces]
    training_set = [dataset[i] for i in training_indeces]

    return training_set, eval_set, timestamps


def split_sets(dataset, satFile_pattern, eval_size = 24, timesensitive = True):
    """
    splits a set of data into an training and an evaluation set
    """

    eval_indeces = []
    timestamps = []
    if(timesensitive):
        if(eval_size % 24 != 0):
            raise ValueError("When using timesensitive splitting eval_size must be multiple of 24")
        timesorted = [[] for _ in range(24)]
        for i in range(len(dataset)):
            sat_id = satFile_pattern.match(os.path.basename(dataset[i][0]))
            if(sat_id):
                timestamp = sat_id.group(1)
                timestamps.append(timestamp)
                hour = int(timestamp[-4:-2])
                timesorted[hour].append(i)
        n = int(eval_size/24)
        for h in range(24):
            eval_indeces += random.sample(timesorted[h], n)
    else:
        eval_indeces = random.sample(range(len(dataset)), eval_size)

    training_indeces = [i for i in range(len(dataset)) if i not in eval_indeces]
    eval_set = [dataset[i] for i in eval_indeces]
    training_set = [dataset[i] for i in training_indeces]

    return training_set, eval_set, timestamps



def get_label_name(sat_file, sat_file_structure, lab_file_structure, timestamp_length):
    timestamp = get_timestamp(sat_file, sat_file_structure, timestamp_length)
    label_file = lab_file_structure.replace("TIMESTAMP", timestamp)
    name, ext = os.path.splitext(label_file)
    label_file = name + "_predicted" + ext
    return label_file




 

def write_NETCDF(data, filename):
    """
    writes xarray dataset to NETCDF file
    """
    data.to_netcdf(path=filename, mode='w')


def check_nwcsaf_version(labels):
    """
    checks if cloud type labels are mapped by the 2013-netcdf standard

    uses occurences in layers 16-19 (only in use at 2013 standard)
    and 7,9,11,13 (only in use at 2018 standard) 
    """
    high_sum = odd_sum = 0
    for i in range(16,20):
        high_sum += (labels == i).sum()
    for i in range(7,14,2):
        odd_sum = (labels == i).sum()

    if (high_sum > 0 and odd_sum == 0):
        return 'v2013'
    if (high_sum == 0 and odd_sum > 0):
        return 'v2018'
    return None



def switch_nwcsaf_version(labels, target_version, input_version = None):
    """
    maps netcdf cloud types from the 2013 standard to the 2018 standard
    """
    if (input_version is None):
        input_version = check_nwcsaf_version(labels)
    if (target_version == input_version):
        return labels
    if (target_version == 'v2018'):
        return switch_2018(labels)
    if (target_version == 'v2013'):
        return switch_2013(labels)



def definde_NWCSAF_variables(missing_labels = None):
    ct_colors = ['#007800', '#000000','#fabefa','#dca0dc',
            '#ff6400', '#ffb400', '#f0f000', '#d7d796',
            '#e6e6e6',  '#c800c8','#0050d7', '#00b4e6',
            '#00f0f0', '#5ac8a0', ]

    ct_indices = [ 1.5, 2.5, 3.5, 4.5, 
               5.5, 6.5, 7.5, 8.5, 
               9.5, 10.5, 11.5, 12.5,
               13.5, 14.5, 15.5]

    ct_labels = ['land', 'sea', 'snow', 'sea ice', 
                 'very low', 'low', 'middle', 'high opaque', 
                 'very high opaque', 'fractional', 'semi. thin', 'semi. mod. thick', 
                 'semi. thick', 'semi. above low','semi. above snow']

    if(missing_labels is not None):
        mis_ind = [ct_labels.index(ml) for ml in missing_labels]
        for ind in sorted(mis_ind, reverse=True):
            for ct_list in [ct_colors, ct_labels, ct_indices]:
                del ct_list[ind]
        

    return ct_colors, ct_indices, ct_labels


def translate_mergeList(merge_list):
    _, indices, labels = definde_NWCSAF_variables()
    intLabel = [int(i) for i in indices]
    merge_ints = []
    for merge_pair in merge_list:
        l1 = labels.index(merge_pair[0]) 
        l2 = labels.index(merge_pair[1])
        merge_ints.append([intLabel[l1],intLabel[l2]])
    return merge_ints


def merge_labels(labels, merge_list):
    merge_ints = translate_mergeList(merge_list)
    for merge_pair in merge_ints:
        labels[labels == merge_pair[1]] = merge_pair[0]


def switch_2018(labels):
    """
    maps netcdf cloud types from the 2013 standard to the 2018 standard
    """
    labels[labels ==  6.0] = 5.0 # very low clouds
    labels[labels ==  8.0] = 6.0 # low clouds
    labels[labels == 10.0] = 7.0 # middle clouds
    labels[labels == 12.0] = 8.0 # high opaque clouds
    labels[labels == 14.0] = 9.0 # very high opaque clouds
    labels[labels == 19.0] = 10.0 # fractional clouds
    labels[labels == 15.0] = 11.0 # high semitransparent thin clouds
    labels[labels == 16.0] = 12.0 # high semitransparent moderatly thick clouds
    labels[labels == 17.0] = 13.0 # high semitransparent thick clouds
    labels[labels == 18.0] = 14.0 # high semitransparent above low or medium clouds
    # missing: 15:  High semitransparent above snow/ice
    return labels


def switch_2013(labels):
    """
    maps netcdf cloud types from the 2018 standard to the 2013 standard
    """
    labels[labels == 15.0] = 18.0 # high semitransparent above snow/ice
    labels[labels == 14.0] = 18.0 # high semitransparent above low or medium clouds
    labels[labels == 13.0] = 17.0 # high semitransparent thick clouds
    labels[labels == 12.0] = 16.0 # high semitransparent moderatly thick clouds
    labels[labels == 11.0] = 15.0 # high semitransparent thin clouds
    labels[labels == 10.0] = 19.0 # fractional clouds
    labels[labels ==  9.0] = 14.0 # very high opaque clouds
    labels[labels ==  8.0] = 12.0 # high opaque clouds
    labels[labels ==  7.0] = 10.0 # middle clouds
    labels[labels ==  6.0] = 8.0 # low clouds
    labels[labels ==  5.0] = 6.0 # very low clouds

    return labels


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
    
    return "Correctly identified {:.2f} %".format(100*correct/total)