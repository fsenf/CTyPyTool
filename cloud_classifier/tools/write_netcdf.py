import xarray as xr
import pathlib
import os



def get_reference_filepath(project_path):
    return os.path.join(project_path, "data", "label_reference.nc")



def create_reference_file(project_path, reference, cloudtype_channel):
    ref_path = get_reference_filepath(project_path)
    if pathlib.Path(ref_path).is_file():  # check if already exists
        return

    data = xr.open_dataset(reference)
    for key in data.keys():
        if(not key == cloudtype_channel):
            data = data.drop(key)
    data.to_netcdf(path=ref_path, mode ='w')
