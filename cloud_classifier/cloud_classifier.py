import tools.io as io
import tools.training as tr
import xarray as xr

import importlib
importlib.reload(tools.io)


class cloud_classifier:
    """
    Trainable Classifier for cloud type prediction from satelite data


    Attributes
    ----------
    training_sets : list
        List of training sets


    Methods
    -------
    add_training_sets(filename_data, filename_labels)
        Reads set of satelite data and according labels and adds it into the classifier
    
    create_trainig_vectors(n)
        Creates training vectors from all added trainig sets

    """


    def __init__(self):
        self.training_sets = []
        self.trainig_vectors = []
        self.mask = None
        self.mask_indices = None
        self.trainig_labels = []
        self.lons = None
        self.lats = None









    def add_training_set(self, filename_data, filename_labels):
        """
        Reads set of satelite data and according labels and adds it into the classifier
        
        Parameters
        ----------
        filename_data : string
            Filename of the mask-data
            
        filename_labels : string
            Filename of the label dataset

        """
        self.training_sets.append([io.read_training_sets(filename_data, filename_labels)])
        return




    def create_trainig_vectors(self, n):
        """
        Creates training vectors from all added trainig sets

        Parameters
        ----------
        n : int
            Number of samples taken for each time value for each training set
        
        """
        if(not self.training_sets):
            print("No training data added")
            return

        if(not self.mask_indices):
            print("No mask given, using complete data set")
            # get all non-nan indices from 'CT' layer of label-data 
            #from first training se at first time slot
            self.mask_indices = np.where(~np.isnan(training_sets[0][1]['CT'][0])

        # get 'n' random samples
        selection = np.array(random.sample(list(zip(self.mask_indices[0],self.mask_indices[1])),n))
        # adjust selection shape (n,2) --> (2,n)
        s = selection.shape
        selection = selection.flatten().reshape(s[1],s[0],order = 'F')   

        # sample all sets

        for t_set in self.training_sets:
            for hour in range(24):
                d,l = tr.extract_trainig_data(t_set[0], t_set[1], selection, hour)

        return





    def add_h5mask(self, filename, selected_mask = None):
        """
        Reads mask-data from h5 file and sets mask for the classifier if specified
        
        Parameters
        ----------
        filename : string
            Filename of the mask-data
            
        mask : string
            (Optional) Name of mask to be used

        """
        if(not training_sets):
            print("Trainig data must be added before setting mask!")
            return

        # get dims and coords from training_sets
        dims = ['rows', 'cols']
        coords = {'lat': training_sets[0].coords['lat'], 'lon':training_sets[0].coords['lon']}
        self.mask = io.read_h5mask(filename, dims, coords)
        # get mask indices if parameter given
        if (selected_mask):
            self.mask_indices = np.where(self.mask["mediterranean_mask"])
        return






    ##### TODO ################
    def clean_masked_indices(self):
        """
        Checks if for the selected mask indices there are NAN values in the dataset(s).

        Goes through all added datasets, and removes all indices from a selected mask, if
        those indices point to NAN values for any time value.
        Can take some time
        """
        for td  in training_sets:
            for i in range(24):
                a,b = extract_learning_data(sat_data, label_data, clean_indeces, hour = i)
                #print(b.shape)
                _, _, clean_indeces = io.clean_data(a,b,clean_indeces)


