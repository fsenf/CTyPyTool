import cloud_project
import parameter_handler
import tools.file_handling as fh
import cloud_classifier
import cloud_plotter
import os
import tools.confusion as conf

import importlib

importlib.reload(cloud_plotter)


class evaluator(cloud_project.cloud_project):
    def __init__(self, project_path=None):

        super().__init__(project_path)
        self.cloud_class = cloud_classifier.cloud_classifier(project_path=project_path)
        self.plotter = cloud_plotter.cloud_plotter()

    def copy_filelists(self, source_project):
        labels_tmp = self.filelists["label_files"]
        self.param_handler.load_filelists(source_project)
        self.filelists[
            "label_files"
        ] = labels_tmp  # don't copy project specific labels safe space
        self.param_handler.save_filelists(self.project_path)
        print("Filelist copied from " + source_project)

    def create_split_trainingset(self, eval_size=24, timesensitive=True):
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

    def create_evaluation_data(self):
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
        cmp_targets=None,
        plot_titles=None,
        show=True,
        verbose=True,
    ):
        self.plotter.load_project(self.project_path)
        self.load_project_data()

        for i in range(len(self.filelists["label_files"])):
            label_file = self.filelists["label_files"][i]
            truth_file = self.filelists["evaluation_sets"][i][1]
            timestamp = self.filelists["eval_timestamps"][i]
            if correlation:
                self.save_coorMatrix(
                    label_file=label_file,
                    truth_file=truth_file,
                    timestamp=timestamp,
                    verbose=verbose,
                    show=show,
                )
            if comparison:
                self.save_comparePlot(
                    label_file=label_file,
                    truth_file=truth_file,
                    timestamp=timestamp,
                    compare_projects=cmp_targets,
                    plot_titles=plot_titles,
                    verbose=verbose,
                    show=show,
                )
            if probabilities:
                self.save_probasPlot(
                    label_file=label_file,
                    truth_file=truth_file,
                    timestamp=timestamp,
                    plot_titles=plot_titles,
                    verbose=verbose,
                    show=show,
                )

    def save_comparePlot(
        self,
        label_file,
        truth_file,
        timestamp,
        compare_projects=None,
        plot_titles=None,
        verbose=True,
        show=True,
    ):
        if compare_projects is None:
            print("No projects for comparison given!")
            return
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
            georef_file=self.georef_file,
            reduce_to_mask=True,
            plot_titles=plot_titles,
            hour=hour,
            save_file=path,
            show=show,
        )
        if verbose:
            print("Comparison Plot saved as " + filename)

    def save_probasPlot(
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

    def save_coorMatrix(
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
            label_data = self.plotter.get_plotable_data(
                data_file=label_file, reduce_to_mask=True, get_coords=False
            )
        if truth_data is None:
            truth_data = self.plotter.get_plotable_data(
                data_file=truth_file, reduce_to_mask=True, get_coords=False
            )

        path = os.path.join(self.project_path, "plots", "Coocurrence", filename)
        fh.create_subfolders(path)

        conf.plot_coocurrence_matrix(
            label_data, truth_data, normalize=normalize, save_file=path, show=show
        )
        if verbose:
            print("Correlation Matrix saved as", filename)

    def get_overallCoocurrence(self, show=False):
        all_labels, all_truth = [], []
        for i in range(len(self.label_files)):
            label_file = self.label_files[i]
            truth_file = self.evaluation_sets[i][1]
            all_labels.append(
                self.plotter.get_plotable_data(data_file=label_file, get_coords=False)
            )
            all_truth.append(
                self.plotter.get_plotable_data(data_file=truth_file, get_coords=False)
            )
        all_labels, all_truth = fh.clean_eval_data(all_labels, all_truth)

        self.save_coorMatrix(
            label_data=all_labels,
            truth_data=all_truth,
            filename="Overall_CoocurrenceMatrix.png",
            show=show,
        )
