#import tools.io as io
import tools.training as tr
import tools.plotting as pl

import xarray as xr
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
import time

import importlib
importlib.reload(tr)
importlib.reload(pl)






class cloud_classifier:
    """
    Trainable Classifier for cloud type prediction from satelite data.



    Methods
    -------
    add_training_sets(filename_data, filename_labels)
        Reads set of satelite data and according labels and adds it into the classifier
    
    create_trainig_set(n)
        Creates training vectors from all added trainig sets

    add_h5mask(filename, selected_mask = None)
        Reads mask-data from h5 file and sets mask for the classifier if specified

    """


    def __init__(self):
        self.training_sets = []
        self.training_vectors = None
        self.training_labels = None
        self.masked_indices = None

        self.pred_vectors = None
        self.pred_labels = None
        self.pred_indices = None
        self.pred_filename = None

        self.cl = None



    def set_mask(self, filename, selected_mask):
        """
        Sets mask for the classifier

        Enables the classefier to only use data from certain regions.
        
        Parameters
        ----------
        filename : string
            Filename of the mask-data
            
        selected_mask : string
            Name of mask to be used

        """    
        self.masked_indices = tr.get_mask_indices(filename, selected_mask)
       


    def add_training_set(self, filename_data, filename_labels):
        """
        Takes set of satelite data and according labels and adds it to the classifier
        
        Parameters
        ----------
        filename_data : string
            Filename of the mask-data
            
        filename_labels : string
            Filename of the label dataset

        """
        self.training_sets.append([filename_data, filename_labels])
        #self.training_sets.append(io.read_training_set(filename_data, filename_labels))
        return



    def create_training_vectors(self, n = 100, hours=range(24), cDV=True, kOV=True):
        """
        Creates sets of training vectors and labels from all added trainig sets

        Parameters
        ----------
        n : int
            Number of samples taken for each time value for each training set

        hour : list
            hours from which vectors are cerated

        cDV: bool
            (Optional) Calculate inter-value-difference-vectors for use as training vectors. Default True.

        kOV : bool
            (Optional) When using difference vectors, also keep original absolut values as 
            first entries in vector. Default False.
        
        """
        if(self.training_sets is None):
            print("No training data added.")
            return

        ##### get training data by sampling all sets at all hours
        self.training_vectors = None
        self.training_labels = np.array([])
        for t_set in self.training_sets:
            v,l = tr.sample_training_set(t_set[0], t_set[1], n, hours, 
                                        self.masked_indices, cDV, kOV)
            if (self.training_vectors is None):
                self.training_vectors = v
            else:
                self.training_vectors = np.append(self.training_vectors, v, axis = 0)
            self.training_labels = np.append(self.training_labels, l, axis = 0)


    def train_tree_classifier(self, max_depth = 20):
        """
        Trains the classifier using previously created training_vectors
        
        Parameters
        ----------
        m_depth : int
            Maximal depth of the decision tree
        """

        self.cl = tree.DecisionTreeClassifier( max_depth = max_depth )

        self.cl.fit(self.training_vectors, self.training_labels)



    def create_test_vectors(self, filename, hour = 0, cDV=True, kOV=True):
        """
        Creates a complete set of vectors from a data set to be used for predicting labels.
        
        Parameters
        ----------
        filename : string
            Filename of the sattelit data set

        hour : int
            The hour of the dataset that is used for vector creation

        cDV: bool
            (Optional) Calculate inter-value-difference-vectors for use as training vectors. Default True.

        kOV : bool
            (Optional) When using difference vectors, also keep original absolut values as 
            first entries in vector. Default False.
        """
        # delete previous labels since they depend on pred_indices
        self.pred_labels = None
        # store filename
        self.pred_filename = filename
        # create vectors for classifier
        self.pred_vectors, self.pred_indices = tr.create_test_vectors(filename,
                                            hour, self.masked_indices, cDV, kOV)


    def predict_labels(self):
        """
        Predicts the labels if a corresponding set of vectors has been created.
        """
        if(self.cl is None):
            print("No classifer trained or loaded")
            return
        if(self.pred_vectors is None):
            print("No test vectors created or added")
            return

        self.pred_labels =  self.cl.predict(self.pred_vectors)

    def plot_labels(self):
        """
        Plots predicted labels
        """
        if (self.pred_labels is None or self.pred_filename is None
                or self.pred_indices is None):
            print("Unsufficant data for plotting labels")
            return
        data = tr.imbed_data(self.pred_labels, self.pred_indices, self.pred_filename)
        pl.plot_data(data)



    def save_labels(self, filename):
        """
        Saves predicted labels
        """
        if (self.pred_labels is None or self.pred_filename is None
                or self.pred_indices is None):
            print("Unsufficant data for saving labels")
            return  
        data = tr.imbed_data(self.pred_labels, self.pred_indices, self.pred_filename)
        tr.write_NETCDF(data, filename)
        
    def evaluate(self, filename_data, filename_labels, hour = 0, cDV=True, kOV=True):
        """
        Predicts the labels for a given satelite data set and evaluates them with.
        
        Parameters
        ----------
        filename_data : string
            Filename of the sattelit data set

        filename_labels : string
            The data of the corresponding labels, if given  

        hour : int
            0-23, hour of the day at which the data set is read

        cDV: bool
            (Optional) Calculate inter-value-difference-vectors for use as training vectors. Default True.

        kOV : bool
            (Optional) When using difference vectors, also keep original absolut values as 
            first entries in vector. Default False.
        """
        if(self.cl is None):
            print("No classifer trained or loaded")
            return
        self.create_test_vectors(filename_data, hour, cDV, kOV)
        self.predict_labels()
        org_labels = tr.exctract_labels_fromFile(filename_labels, self.pred_indices, hour)

        correct = np.sum(self.pred_labels == org_labels)
        total = len(org_labels)
        print("Correctly identified %i out of %i labels! \nPositve rate is: %f" % (correct, total, correct/total))


    def save_classifier(self, filename):
        """
        Predicts the labels for a given satelite data set and evaluates them with.
        
        Parameters
        ----------
        filename : string
            Name if the file the classifier is saved under
        """
        dump(self.cl, filename)

    def load_classifier(self, filename):
        """
        Predicts the labels for a given satelite data set and evaluates them with.
        
        Parameters
        ----------
        filename : string
            Name if the file the classifier is loaded from
        """
        self.cl = load(filename)
