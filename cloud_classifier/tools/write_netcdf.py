import xarray as xr
import pathlib



def get_reference_filepath(project_path):
    return pathlib.Path.join(project_path, "data", "label_reference.nc")



def create_reference_file(project_path, param_handler):
    ref_path = get_reference_filepath(project_path)
    if pathlib.Path(ref_path).is_file():  # check if already exists
        return

    training_sets = param_handler.filelists["training_sets"]
    if (not training_sets):
        raise RuntimeError("Can not create reference file. No training data added")
    input_file = training_sets[0][1]
    data = xr.open_dataset(input_file)
    for key in data.keys():
        if(not key == param_handler.parameters["cloudtype_channel"]):
            data = data.drop(key)
    data.to_netcdf(path=ref_path, mode ='w')
