from joblib import dump, load

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import os


class cloud_trainer:

    """
    Cloud trainer class that facilitates the training of a classifier with previously
    extracted training data as well as the predicting of labels from satelite data.

    Attributes
    ----------
    classifier :
        Classifier that is trained by the cloud_trainer.
    """

    def train_classifier(self, vectors, labels, params, verbose=True):
        """
        Trains the classifier using previously created vectors

        Parameters
        ----------
        vectors : numpy array
            Array of trainig vectors extracted from satelite data.
        labels : numpy array
            Array of labels corrosponding to the training vectors
        params : dicitonary
            Description
        verbose : bool, optional
            If True, the function will give detailed command line output.
        """

        classifier_type = params["classifier_type"]

        if classifier_type == "Tree":
            print("Training Tree Classifier")
            self.classifier = make_pipeline(
                StandardScaler(),
                DecisionTreeClassifier(
                    max_depth=params["max_depth"], ccp_alpha=params["ccp_alpha"]
                ),
            )
        elif classifier_type == "Forest":
            print("Training Forest Classifier")
            self.classifier = make_pipeline(
                StandardScaler(),
                RandomForestClassifier(
                    n_estimators=params["n_estimators"],
                    max_depth=params["max_depth"],
                    ccp_alpha=params["ccp_alpha"],
                    min_samples_split=params["min_samples_split"],
                    max_features=params["max_features"],
                ),
            )
        elif classifier_type == "SVM":
            print("Training SVM")
            self.classifier = make_pipeline(StandardScaler(), SVC(gamma="auto"))

        else:
            raise ValueError("Classifier type not valid")

        self.classifier.fit(vectors, labels)
        if verbose:
            print("Classifier trained!")

    def predict_labels(self, vectors):
        """
        Predicts the labels from a corresponding set of input vectors.

        Parameters
        ----------
        vectors : numpy array
            Array of input vectors.

        Returns
        -------
        numpy array
            Array of predicted labels
        """
        if self.classifier is None:
            raise Exception("No classifer trained or loaded")
        return self.classifier.predict(vectors)

    def get_forest_proabilties(self, vectors):
        """
        Predicts the proabilities of class memebership for all target classes.
        Only applicable if classifier is of Random Forest type

        Parameters
        ----------
        vectors : numpy array
            Array of input vectors.

        Returns
        -------
        numpy array
            Array of predicted probability values.
        """
        if self.classifier is None:
            raise Exception("No classifer trained or loaded")
        return self.classifier.predict_proba(vectors)

    def save_classifier(self, project_path, verbose=True):
        """
        Saves Classifier

        Parameters
        ----------
        project_path : string
            Path of the project folder.
        verbose : bool, optional
            If True, the function will give detailed command line output.
        """
        filename = os.path.join(project_path, "data", "classifier")
        dump(self.classifier, filename)
        if verbose:
            print("Classifier saved!")

    def load_classifier(self, project_path, verbose=True):
        """
        Loads classifer from project folder.

        Parameters
        ----------
        project_path : string
            Path of the project folder.
        verbose : bool, optional
            If True, the function will give detailed command line output.

        """
        filename = os.path.join(project_path, "data", "classifier")
        self.classifier = load(filename)
        if verbose:
            print("Classifier loaded!")


"""
    def fit_feature_selection(self, vectors, labels, k=20):
        self.feat_select = SelectKBest(k=k).fit(vectors, labels)


    def apply_feature_selection(self, vectors):
        if self.feat_select is None:
            print("No feature selection fitted")
            return
        return self.feat_select.transform(vectors)
"""
