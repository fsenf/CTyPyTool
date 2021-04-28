#import tools.io as io
import tools.training as tr
import tools.plotting as pl

import xarray as xr
import numpy as np

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2

from joblib import dump, load
import time

import importlib
importlib.reload(tr)
importlib.reload(pl)






class cloud_classifier:
    """
    Trainable Classifier for cloud cl_type prediction from satelite data.



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
        self.feat_select = None

        ### paramaeters
        self.cDV = False
        self.kOV = False
        self.n = 100
        self.hours=range(24)
        self.cl_type = "Tree"


    def set_paremeters(self, n=100, hours=range(24), cDV=True, kOV=True, cl_type = "Tree"):
        self.cDV = cDV
        self.kOV = kOV
        self.n = n
        self.hours=hours
        self.cl_type = cl_type



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
        return



    def create_training_vectors(self):
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
            v,l = tr.sample_training_set(t_set[0], t_set[1], self.n, self.hours, 
                                        self.masked_indices, self.cDV, self.kOV)
            if (self.training_vectors is None):
                self.training_vectors = v
            else:
                self.training_vectors = np.append(self.training_vectors, v, axis = 0)
            self.training_labels = np.append(self.training_labels, l, axis = 0)


    def fit_feature_selection(self, k = 20):
        if(self.training_vectors is None or self.training_labels is None):
            print("No training vectors ceated")
            return
        self.feat_select = SelectKBest(chi2, k=k).fit(self.training_vectors, self.training_labels)

    def apply_feature_selection(self, vectors = None):
        return

    def train_tree_classifier(self, max_depth = None, ccp_alpha = None, training_vectors = None, training_labels = None):
        """
        Trains the classifier using previously created training_vectors
        
        Parameters
        ----------
        m_depth : int
            Maximal depth of the decision tree
        """
        if(self.cl_type == "Tree"):
            self.cl = tree.DecisionTreeClassifier(max_depth = max_depth, ccp_alpha=ccp_alpha)
        elif(self.cl_type == "Forest"): 
            self.cl = RandomForestClassifier( n_estimators = 75, max_depth = max_depth, ccp_alpha=ccp_alpha)


        if (training_vectors is None or training_labels is None):
            self.cl.fit(self.training_vectors, self.training_labels)
        else:
            self.cl.fit(training_vectors, training_labels)



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
                            self.hour, self.masked_indices, self.cDV, self.kOV)


    def predict_labels(self):
        """
        Predicts the labels if a corresponding set of input vectors has been created.
        """
        if(self.cl is None):
            print("No classifer trained or loaded")
            return
        if(self.pred_vectors is None):
            print("No input vectors created or added")
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



    ##################################################################################
    #################         Evaluation
    ##################################################################################
        
    # def create_evaluation_set(self, filename = None):
    #     """
    #     Creates a set of training and test vectors for evaluation
        
    #     Parameters
    #     ----------
    #     self : cloud_classifier
    #         Object of cloud classifier cl_type
    #     filename : string
    #         (Optional) Filename for saving dataset

    #     """

    #     if(self.training_sets is None):
    #         print("No training data added.")
    #     return

    #     self.create_training_vectors()
    #     train_v, test_v, train_l, test_l = train_test_split(
    #                         self.training_vectors,self.training_labels, random_state=0)

    #     dump([train_v, train_l], filename)




    def evaluate_parameters(self, max_depth = None, ccp_alpha = 0, verbose = False):
        """
        Evaluates the given parameters over a set of training vectors

        Training vectors are split into test and trainig set
        """
        if(self.training_vectors is None or self.training_labels is None):
            print("No training vectors ceated")
            return
        train_v, test_v, train_l, test_l = train_test_split(
                                self.training_vectors,self.training_labels, random_state=0)

        self.train_tree_classifier(max_depth, ccp_alpha, train_v, train_l)

        #print(self.cl.score(test_v, test_l))
        pred_l = self.cl.predict(test_v)

        correct = np.sum(pred_l == test_l)
        total = len(pred_l)
        if(verbose):
            print("Correctly identified %i out of %i labels! \nPositve rate is: %f" % (correct, total, correct/total))
        return(self.cl.score(test_v, test_l))
        

        
    def evaluate_classifier(self, filename_data, filename_labels, hour = 0):
        """
        Evaluates an already trained classifier with a new set of data and labels
        
        Parameters
        ----------
        filename_data : string
            Filename of the sattelit data set

        filename_labels : string
            The data of the corresponding labels, if given  

        hour : int
            0-23, hour of the day at which the data sets are read
        """
        if(self.cl is None):
            print("No classifer trained or loaded")
            return
        self.create_test_vectors(filename_data, hour, self.cDV, self.kOV)
        self.predict_labels()
        org_labels = tr.exctract_labels_fromFile(filename_labels, self.pred_indices, hour)

        correct = np.sum(self.pred_labels == org_labels)
        total = len(org_labels)
        print("Correctly identified %i out of %i labels! \nPositve rate is: %f" % (correct, total, correct/total))
  

    ##################################################################################
    #################         Saving and Loading parts of the data
    ##################################################################################

    def export_labels(self, filename):
        """
        Saves predicted labels as netcdf file

        Parameters
        ----------
        filename : string
            Name of the file in which the labels will be written
        """
        if (self.pred_labels is None or self.pred_filename is None
                or self.pred_indices is None):
            print("Unsufficant data for saving labels")
            return  
        data = tr.imbed_data(self.pred_labels, self.pred_indices, self.pred_filename)
        tr.write_NETCDF(data, filename)
    

    def save_classifier(self, filename):
        """
        Saves Classifier

        Parameters
        ----------
        filename : string
            Name of the file into which the classifier is saved.
        """
        dump(self.cl, filename)


    def load_classifier(self, filename):
        """
        Loads classifer        
        Parameters
        ----------
        filename : string
            Name if the file the classifier is loaded from.
        """
        self.cl = load(filename)


    def save_training_vector_set(self, filename):
        """
        Saves a set of already created training vectors

        ----------
        filename : string
            Name of the file into which the vector set is saved.
        """
        dump([self.training_vectors, self.training_labels], filename)


    def load_training_vector_set(self, filename):
        """
        Loads a set of already created training vectors
        
        Parameters
        ----------
        filename : string
            Name if the file the vector set is loaded from.
        """
        self.training_vectors, self.training_labels = load(filename)



