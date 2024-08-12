import os

from . import cloud_project
from . import parameter_handler
from . import cloud_classifier
from . import cloud_plotter

from .tools import confusion as conf
from .tools import file_handling as fh



class evaluator(cloud_project.cloud_project):

    """
    Class that provides functionality for comparing classifiers with different approaches
    or training parameters with each other.

    Attributes
    ----------
    cloud_class : cloud_classifer
        Cloud classifer instance used by this class to train classifier and predict evaluation labels.
    plotter : cloud_plotter
        Cloud plotter instance used to plot results.
    """

    def __init__(self, project_path=None):

        super().__init__(project_path)
        self.cloud_class = cloud_classifier.cloud_classifier(project_path=project_path)
        self.plotter = cloud_plotter.cloud_plotter()

    def create_split_trainingset(self, eval_size=24, timesensitive=True):
        """
        Creates a filelist from the files in the data_source_folder specified in the
        project settings and splits the filelist into a training and an evaluation
        set.

        Parameters
        ----------
        eval_size : int, optional
            Default value 24. Number of datasets in the evaluation set. Must be multiple
            of 24 if timesensitive is True
        timesensitive : bool, optional
            If True, the evaluation set is equally distributed over all hours of the day.
            Requires eval_size to be multiple of 24.
        """
        self.load_project_data()
        satFile_pattern = fh.get_filename_pattern(
            self.params["sat_file_structure"], self.params["timestamp_length"]
        )
        labFile_pattern = fh.get_filename_pattern(
            self.params["label_file_structure"], self.params["timestamp_length"]
        )
        datasets = fh.generate_filelist_from_folder(
            self.params["data_source_folder"], satFile_pattern, labFile_pattern
        )

        training_sets, evaluation_sets, timestamps = fh.split_sets(
            datasets, satFile_pattern, eval_size=eval_size, timesensitive=timesensitive
        )

        self.param_handler.set_filelists(
            training_sets=training_sets,
            evaluation_sets=evaluation_sets,
            input_files=[s[0] for s in evaluation_sets],
            eval_timestamps=timestamps,
        )

        self.param_handler.save_filelists(self.project_path)

    def copy_evaluation_split(self, source_project):
        """
        Copies the distribution into training and evaluation sets from another project.

        Parameters
        ----------
        source_project : str
            Path to project from which the data is copied.
        """
        labels_tmp = self.filelists["label_files"]
        self.param_handler.load_filelists(source_project)
        self.filelists[
            "label_files"
        ] = labels_tmp  # don't copy project specific labels safe space
        self.param_handler.save_filelists(self.project_path)
        print("Filelist copied from " + source_project)

    def create_evaluation_data(self):
        """
        Trains a classifier according to the project parameters and predicts labels for all
        files in the evaluation set.
        """
        self.cloud_class.load_project(self.project_path)
        print("Trainig evaluation classifier")
        self.cloud_class.run_training_pipeline(create_filelist=False)
        print("Prediciting evaluation labels")
        self.cloud_class.run_prediction_pipeline(create_filelist=False)

    def create_evaluation_plots(
        self,
        correlation=False,
        probabilities=False,
        comparison=False,
        overallCorrelation=False,
        cmp_targets=[],
        plot_titles=[],
        show=True,
        verbose=True,
    ):
        """
        Creates evaluation plots for all the predicted data. The data can be plotted in
        comparison to the ground truth or other classifiers predictions. Plots are saved
        in the project folder.

        Parameters
        ----------
        correlation : bool, optional
            If True, Coocurrence Matrices of the predicted data and the ground truth will be plotted.
        probabilities : bool, optional
            If True, plots containing certainity of prediciton as subplot will be created.
        comparison : bool, optional
            If True, comparisons with other classifiers will be plotted.
            Requires cmp_targets to contain other classifier projects.
        overallCorrelation : bool, optional
            If True, overall Coocurrence Matrix over all evaluation sets will be calculated an plotted.
        cmp_targets : list, optional
            List of filepaths of cloud classifier projects for comparison plots.
        plot_titles : list, optional
            List of strings with Titles of the subplots.
        show : bool, optional
            If True plot will be displayed in additon to be saved in the project folder
        verbose : bool, optional
            Description
        """
        self.plotter.load_project(self.project_path)
        self.load_project_data()

        for i in range(len(self.filelists["label_files"])):
            label_file = self.filelists["label_files"][i]
            truth_file = self.filelists["evaluation_sets"][i][1]
            timestamp = self.filelists["eval_timestamps"][i]
            if correlation:
                self.__save_coorMatrix(
                    label_file=label_file,
                    truth_file=truth_file,
                    timestamp=timestamp,
                    verbose=verbose,
                    show=show,
                )
            if comparison:
                self.__save_comparePlot(
                    label_file=label_file,
                    truth_file=truth_file,
                    timestamp=timestamp,
                    compare_projects=cmp_targets,
                    plot_titles=plot_titles,
                    verbose=verbose,
                    show=show,
                )
            if probabilities:
                if not plot_titles:
                    plot_titles = ["Probability Score", "Prediction"]
                self.__save_probasPlot(
                    label_file=label_file,
                    truth_file=truth_file,
                    timestamp=timestamp,
                    plot_titles=plot_titles,
                    verbose=verbose,
                    show=show,
                )
        if overallCorrelation:
            self.__get_overallCoocurrence(show=show)

    def __save_comparePlot(
        self,
        label_file,
        truth_file,
        timestamp,
        compare_projects=[],
        plot_titles=None,
        verbose=True,
        show=True,
    ):

        all_files = [label_file]
        filename = os.path.split(label_file)[1]
        for proj_path in compare_projects:
            path = os.path.join(proj_path, "labels", filename)
            all_files.append(path)

        filename = timestamp + "_ComparisonPlot.png"
        path = os.path.join(self.project_path, "plots", "Comparisons", filename)
        fh.create_subfolders(path, self.project_path)

        hour = int(timestamp[-4:-2])
        self.plotter.plot_multiple(
            all_files,
            truth_file,
            georef_file=self.params["georef_file"],
            reduce_to_mask=True,
            plot_titles=plot_titles,
            hour=hour,
            save_file=path,
            show=show,
        )
        if verbose:
            print("Comparison Plot saved as " + filename)

    def __save_probasPlot(
        self,
        label_file,
        truth_file,
        timestamp,
        plot_titles=None,
        verbose=True,
        show=True,
        filename=None,
    ):

        if filename is None:
            filename = timestamp + "_ProbabilityPlot.png"
        path = os.path.join(self.project_path, "plots", "Probabilities", filename)
        fh.create_subfolders(path, self.project_path)

        hour = int(timestamp[-4:-2])
        self.plotter.plot_probas(
            label_file,
            truth_file,
            georef_file=self.params["georef_file"],
            reduce_to_mask=True,
            plot_titles=plot_titles,
            hour=hour,
            save_file=path,
            show=show,
        )
        if verbose:
            print("Probability Plot saved as " + filename)

    def __save_coorMatrix(
        self,
        label_file=None,
        truth_file=None,
        label_data=None,
        truth_data=None,
        timestamp=None,
        filename=None,
        normalize=True,
        verbose=True,
        show=True,
    ):

        if truth_file is None and truth_data is None:
            raise ValueError("'truth_file' or 'truth_data' be specified!")
        if label_file is None and label_data is None:
            raise ValueError("'label_file' or 'label_data' be specified!")
        if filename is None and timestamp is None:
            raise ValueError("'filename' or 'timestamp' be specified!")

        if filename is None:
            filename = timestamp + "_CoocurrenceMatrix.png"
        if label_data is None:
            label_data = self.plotter.get_plottable_data(
                data_file=label_file, reduce_to_mask=True, get_coords=False
            )
        if truth_data is None:
            truth_data = self.plotter.get_plottable_data(
                data_file=truth_file, reduce_to_mask=True, get_coords=False
            )

        path = os.path.join(self.project_path, "plots", "Coocurrence", filename)
        fh.create_subfolders(path)

        conf.plot_coocurrence_matrix(
            label_data, truth_data, normalize=normalize, save_file=path, show=show
        )
        if verbose:
            print("Correlation Matrix saved as", filename)

    def __get_overallCoocurrence(self, show=False):
        all_labels, all_truth = [], []
        for i in range(len(self.filelists["label_files"])):
            label_file = self.filelists["label_files"][i]
            truth_file = self.filelists["evaluation_sets"][i][1]
            all_labels.append(
                self.plotter.get_plottable_data(data_file=label_file, get_coords=False)
            )
            all_truth.append(
                self.plotter.get_plottable_data(data_file=truth_file, get_coords=False)
            )
        all_labels, all_truth = fh.clean_eval_data(all_labels, all_truth)

        self.__save_coorMatrix(
            label_data=all_labels,
            truth_data=all_truth,
            filename="Overall_CoocurrenceMatrix.png",
            show=show,
        )
