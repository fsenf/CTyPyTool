import tools.io
import xarray as xr


class cloud_classifier:
    """
    Trainable Classifier for cloud type prediction from satelite data


    Attributes
    ----------
    training_data : list
        List of training sets


    Methods
    -------
    add_training_data(filename_data, filename_labels)
        Reads data from file and adds it to list

    """

    def __init__(self):
        self.training_data = []
        self.lons = None
        self.lats = None
        self.mask = None

    def add_trainig_data(self, filename_data, filename_labels):
        return



    def read_h5mask(filename, mask = Null):
        """
        Reads mask-data from h5 file and sets mask for the classifier if given   
        
        Parameters
        ----------
        filename : string
            Filename of the mask-data
            
        mask : string
            (Optional) Name of mask to be used

        """
        if(not training_data):
            print("Trainig data must be added before setting mask!")
            return
        dims = ['rows', 'cols']
        coords = {'lat': training_data[0].coords['lat'], 'lon':training_data[0].coords['lon']}
        io.read_mask(filename, dims, coords)
        return