import xarray as xr
import numpy as np
from joblib import dump, load

import sklearn.svm as svm
from sklearn import tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2

import base_class
import importlib
import inspect


from base_class import base_class

class cloud_trainer(base_class):
    """
    Trainable Classifier for cloud classifier_type prediction from satelite data.



    Methods
    -------

    """


    def __init__(self, **kwargs):
        class_variables =  {
            'classifier_type', 
            'max_depth',
            'ccp_alpha',
            'n_estimators',
            'feature_preselection',
            'max_features',
            'min_samples_split',
            }
        
        self.classifier = None      

        super().init_class_variables(class_variables)
        super().__init__( **kwargs)
   



    # def set_default_parameters(self, reset_data = False):
    #     ### paramaeters
    #     self.classifier_type = "Tree"
    #     self.max_depth = 20
    #     self.ccp_alpha = 0
    #     self.feature_preselection = False
    #     self.n_estimators = 75

    #     if (reset_data):
    #         self.pred_labels = None
    #         self.classifier = None
    #         self.feat_select = None


    def fit_feature_selection(self, training_vectors, training_labels, k = 20):
        self.feat_select = SelectKBest(k=k).fit(training_vectors, training_labels)



    def apply_feature_selection(self, vectors):
        if(self.feat_select is None):
            print("No feature selection fitted")
            return
        return self.feat_select.transform(vectors)



    def train_classifier(self, training_vectors, training_labels):
        """
        Trains the classifier using previously created training_vectors
        
        Parameters
        ----------
        m_depth : int
            Maximal depth of the decision tree
        """

        if(self.classifier_type == "Tree"):
            self.classifier = tree.DecisionTreeClassifier(max_depth = self.max_depth, ccp_alpha = self.ccp_alpha)
        elif(self.classifier_type == "Forest"): 
            self.classifier = RandomForestClassifier(n_estimators = self.n_estimators, max_depth = self.max_depth, 
                                                 ccp_alpha = self.ccp_alpha, min_samples_split = self.min_samples_split,
                                                 max_features = self.max_features)
        elif(self.classifier_type == "AdaBoost"):
            svc=SVC(probability=True, kernel='linear')
            self.classifier = AdaBoostClassifier(n_estimators =  self.n_estimators, base_estimator=svc)

        elif(self.classifier_type == "nuSVM"):
            self.classifier = svm.SVC(probability=True, kernel='linear')


        if(training_vectors is None or training_labels is None):
            print("No training data!")
            return

        if(self.feature_preselection and not (self.feat_select is None)):
            training_vectors = self.apply_feature_selection(training_vectors)
        self.classifier.fit(training_vectors, training_labels)




    def predict_labels(self, vectors):
        """
        Predicts the labels if a corresponding set of input vectors has been created.
        """
        if(self.classifier is None):
            print("No classifer trained or loaded")
            return

        if(self.feature_preselection and not (self.feat_select is None)):
            vectors = self.apply_feature_selection(vectors)

        return self.classifier.predict(vectors)



    def get_forest_proabilties(self, vectors):
        if(self.classifier is None):
            print("No classifer trained or loaded")
            return

        if(self.feature_preselection and not (self.feat_select is None)):
            vectors = self.apply_feature_selection(vectors)

        return self.classifier.predict_proba(vectors)





    def evaluate_parameters(self, vectors, labels, verbose = True):
        """
        Evaluates the given parameters over a set of training vectors

        Training vectors are split into test and trainig set
        """

        # save a possible already trained classifier
        tmp = self.classifier
        train_v, test_v, train_l, test_l = train_test_split(vectors, labels, random_state=0)

        self.train_classifier(train_v, train_l)

        pred_l = self.predict_labels(test_v)

        correct = np.sum(pred_l == test_l)
        total = len(pred_l)
        #restore classifier
        self.classifier = tmp
        if(verbose):
            print("Correctly identified %i out of %i labels! \nPositve rate is: %f" % (correct, total, correct/total))
        return(correct/total)
        

        
    def evaluate_classifier(self, vectors, labels):
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
        if(self.classifier is None):
            print("No classifer trained or loaded")
            return
        predicted_labels = self.predict_labels(vectors)

        correct = np.sum(predicted_labels == labels)
        total = len(labels)
        
        print("Correctly identified %i out of %i labels! \nPositve rate is: %f" % (correct, total, correct/total))
        return(correct/total)








    ##################################################################################
    #################         Saving and Loading classifier
    ##################################################################################


    def save_classifier(self, filename):
        """
        Saves Classifier

        Parameters
        ----------
        filename : string
            Name of the file into which the classifier is saved.
        """
        dump(self.classifier, filename)


    def load_classifier(self, filename):
        """
        Loads classifer        
        Parameters
        ----------
        filename : string
            Name if the file the classifier is loaded from.
        """
        self.classifier = load(filename)
