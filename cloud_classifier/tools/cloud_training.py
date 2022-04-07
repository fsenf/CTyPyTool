import numpy as np
from joblib import dump, load

from sklearn import tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler



def fit_feature_selection(training_vectors, training_labels, k = 20):
    self.feat_select = SelectKBest(k=k).fit(training_vectors, training_labels)



def apply_feature_selection(vectors):
    if(self.feat_select is None):
        print("No feature selection fitted")
        return
    return self.feat_select.transform(vectors)



def train_classifier(training_vectors, training_labels, refined = False):
    """
    Trains the classifier using previously created training_vectors
    
    Parameters
    ----------
    m_depth : int
        Maximal depth of the decision tree
    """
    # if(refined):
    #     print("Training Forest Classifier")
    #     self.classifier = RandomForestClassifier(n_estimators = self.n_estimators, max_depth = self.max_depth, 
    #                                          ccp_alpha = self.ccp_alpha, min_samples_split = self.min_samples_split,
    #                                          max_features = self.max_features)

    # if(refined):
    #     print("Training Tree Classifier")
    #     self.classifier = tree.DecisionTreeClassifier(max_depth = self.max_depth, ccp_alpha = self.ccp_alpha)
   

    if(training_vectors is None or training_labels is None):
        raise ValueError("Missing Training data")
        return


    classifier_type = self.classifier_type
    if(refined):
        classifier_type = self.refinment_classifier_type


    if(classifier_type == "Tree"):
        print("Training Tree Classifier")
        self.classifier = tree.DecisionTreeClassifier(max_depth = self.max_depth, ccp_alpha = self.ccp_alpha)
    elif(classifier_type == "Forest"): 
        print("Training Forest Classifier")
        self.classifier = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators = self.n_estimators, max_depth = self.max_depth, 
                                             ccp_alpha = self.ccp_alpha, min_samples_split = self.min_samples_split,
                                             max_features = self.max_features))
    elif(classifier_type == "SVM"): 
        print("Training SVM")
        self.classifier = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    
    else:
        raise ValueError("Classifier type not valid")


    if(self.feature_preselection and not (self.feat_select is None)):
        training_vectors = self.apply_feature_selection(training_vectors)
    self.classifier.fit(training_vectors, training_labels)



def predict_labels(vectors):
    """
    Predicts the labels if a corresponding set of input vectors has been created.
    """
    if(self.classifier is None):
        print("No classifer trained or loaded")
        return

    if(self.feature_preselection and not (self.feat_select is None)):
        vectors = self.apply_feature_selection(vectors)

    return self.classifier.predict(vectors)



def get_forest_proabilties(vectors):
    if(self.classifier is None):
        print("No classifer trained or loaded")
        return

    if(self.feature_preselection and not (self.feat_select is None)):
        vectors = self.apply_feature_selection(vectors)

    return self.classifier.predict_proba(vectors)





def evaluate_parameters(vectors, labels, verbose = True):
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
    

    
def evaluate_classifier(vectors, labels):
    """
    Evaluates an already trained classifier with a new set of data and labels
    
    Parameters
    ----------
    filename_data : string
        Filename of the sattelit data set

    filename_labels : string
        The data of the corresponding labels, if given  

    hour : int
        0-23, hour of the day at wh the data sets are read
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


def save_classifier(filename):
    """
    Saves Classifier

    Parameters
    ----------
    filename : string
        Name of the file into wh the classifier is saved.
    """
    dump(self.classifier, filename)


def load_classifier(filename):
    """
    Loads classifer        
    Parameters
    ----------
    filename : string
        Name if the file the classifier is loaded from.
    """
    self.classifier = load(filename)
