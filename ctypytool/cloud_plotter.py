import numpy as np
import xarray as xr
import warnings

import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs

import tools.data_handling as dh
import tools.training_data as td
import tools.file_handling as fh
import tools.nwcsaf_tools as nwc
import tools.confusion as conf

import cloud_project


import importlib

importlib.reload(nwc)


class cloud_plotter(cloud_project.cloud_project):

    """
    Plotting class building on cloud_project class.
    Provides functionality of preparing and plotting data.
    """

    def plot_data(
        self,
        label_file,
        reduce_to_mask=True,
        extent=[-6, 42, 25, 50],
        cmap="hot",
        mode="label",
        subplot=False,
        pos=None,
        colorbar=False,
        cb_pos=0.95,
    ):
        """
        Basic class for plotting data onto map.

        Parameters
        ----------
        label_file : str
            Filepath of NWCSAF type file from which labels are read.
        reduce_to_mask : bool, optional
            If True, data will be reduced to mask specified in the project settings.
        extent : list , optional
            Extent of the map sections, default shows mediteranian sea.
        cmap : str, optional
            Color mapping of the input data.
        mode : str, optional
            Determines mode of plot, 'label' will show the labels, 'proba' the certainity of prediciton.
        subplot : bool, optional
            If True, plot will be integrated as subplot into larger multi-plot image.
        pos : list, optional
            Position in image if subplot.
        colorbar : bool, optional
            If True, a labbeled colorbar will be added to plot.
        cb_pos : float, optional
            Float between 0 and 1. Relative vertical Position of colorbar in plot.
        """
        if mode == "label":
            ct_colors, ct_indices, ct_labels = nwc.definde_NWCSAF_variables()
            cmap = plt.matplotlib.colors.ListedColormap(ct_colors)
            vmin = 1
            vmax = 15
        elif mode == "proba":
            vmin = 0.0
            vmax = 1.0

        if not subplot:
            plt.figure(figsize=(13, 8))

        if subplot:
            ax = plt.subplot(pos[0], pos[1], pos[2], projection=ccrs.PlateCarree())
        else:
            ax = plt.axes(projection=ccrs.PlateCarree())

        ax.coastlines(resolution="50m")
        ax.set_extent(extent)
        ax.add_feature(cartopy.feature.LAND, edgecolor="black")
        data, x, y = self.get_plottable_data(
            data_file=label_file, reduce_to_mask=reduce_to_mask, mode=mode
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pcm = ax.pcolormesh(x, y, data, cmap=cmap, vmin=vmin, vmax=vmax)

        if colorbar:
            fig = plt.gcf()
            a2 = fig.add_axes([cb_pos, 0.22, 0.015, 0.6])
            cbar = plt.colorbar(pcm, a2)
            if mode == "label":
                cbar.set_ticks(ct_indices)
                cbar.ax.tick_params(labelsize=14)
                cbar.set_ticklabels(ct_labels)

        if subplot:
            return ax, data

    def plot_probas(
        self,
        label_file,
        truth_file=None,
        georef_file=None,
        reduce_to_mask=True,
        plot_corr=False,
        plot_titles=[],
        hour=None,
        save_file=None,
        show=True,
    ):
        """
        Function that plots certainity plot for predicted labels (only for Random Forest)
        next to label data and optionally the ground truth data in a multi-plot image.

        Parameters
        ----------
        label_file : str
            Filepath of NWCSAF type file from which labels are read.
        truth_file : str, optional
            Filepath of NWCSAF type file from which ground truth labels are read.
            If None, ground truth will not be added to plot.
        georef_file : str, optional
            Filepath of georef file from which lat and lon values read.
            If None, georef file from project settings will be used.
        reduce_to_mask : bool, optional
            If True, data will be reduced to mask specified in the project settings.
        plot_titles : list, optional
            Plot titles
        hour : str, optional
            Time of day from the data. If not None this will marked on the image.
        save_file : str, optional
            Filepath for saving the plot. If None, plot will not be saved.
        show : bool, optional
            If True, plot will be displayed to the user.
        """
        gt = truth_file is not None
        length = 2
        if truth_file is not None:
            length += 1

        fig = plt.figure(figsize=(length * 13, 8))
        fig.patch.set_alpha(1)

        if gt:
            pos = [1, length, length]
            ax, truth = self.plot_data(
                truth_file, reduce_to_mask, pos=pos, subplot=True, colorbar=True
            )
            if hour is not None:
                text = "Time: {:02d}:00".format(hour)
                ax.text(10, 22, text, fontsize=16)
            if plot_titles or length > len(plot_titles):
                ax.set_title("Ground Truth", fontsize=20)

        modes = ["proba", "label"]
        cb = [True, not gt]
        cb_p = [0.05, 0.95]
        for i in range(len(modes)):
            pos = [1, length, i + 1]
            ax, data = self.plot_data(
                label_file,
                reduce_to_mask=reduce_to_mask,
                pos=pos,
                subplot=True,
                mode=modes[i],
                colorbar=cb[i],
                cb_pos=cb_p[i],
            )

            if plot_titles and len(plot_titles) > i:
                ax.set_title(plot_titles[i], fontsize=20)

            if gt and i == 1:
                text = fh.get_match_string(data, truth)
                ax.text(10, 22, text, fontsize=16)

        plt.subplots_adjust(wspace=0.05)
        if save_file is not None:
            plt.savefig(save_file, transparent=False)
        if show:
            plt.show()
        plt.close()

    def plot_multiple(
        self,
        label_files,
        truth_file=None,
        georef_file=None,
        reduce_to_mask=True,
        plot_titles=None,
        hour=None,
        save_file=None,
        show=True,
    ):
        """
        Function that plots the data from multiple label files and optionally the ground truth
        data in a multi-plot image.



        Parameters
        ----------
        label_files : list
            List of filepaths of NWCSAF type files from which labels are read.
        truth_file : str, optional
            Filepath of NWCSAF type file from which ground truth labels are read.
            If None, ground truth will not be added to plot.
        georef_file : str, optional
            Filepath of georef file from which lat and lon values read.
            If None, georef file from project settings will be used.
        reduce_to_mask : bool, optional
            If True, data will be reduced to mask specified in the project settings.
        plot_titles : list, optional
            Plot titles
        hour : str, optional
            Time of day from the data. If not None this will marked on the image.
        save_file : str, optional
            Filepath for saving the plot. If None, plot will not be saved.
        show : bool, optional
            If True, plot will be displayed to the user.

        """

        length = len(label_files)
        lab_lenght = length
        if truth_file is not None:
            length += 1
        fig = plt.figure(figsize=(length * 13, 8))
        fig.patch.set_alpha(1)

        # plot ground truth
        if truth_file is not None:
            pos = [1, length, length]
            ax, truth = self.plot_data(
                truth_file,
                reduce_to_mask=reduce_to_mask,
                pos=pos,
                subplot=True,
                colorbar=True,
            )
            if hour is not None:
                text = "Time: {:02d}:00".format(hour)
                ax.text(10, 22, text, fontsize=16)
            if length > len(plot_titles):
                ax.set_title("Ground Truth", fontsize=20)

        # plot labels
        for i in range(lab_lenght):
            pos = [1, length, i + 1]
            ax, data = self.plot_data(
                label_files[i], reduce_to_mask, pos=pos, subplot=True
            )

            if plot_titles is not None and i < len(plot_titles):
                ax.set_title(plot_titles[i], fontsize=20)

            if truth_file is not None and i < lab_lenght:
                text = fh.get_match_string(data, truth)
                ax.text(10, 22, text, fontsize=16)

        plt.subplots_adjust(wspace=0.05)
        if save_file is not None:
            plt.savefig(save_file, transparent=False)
        if show:
            plt.show()
        plt.close()

    def get_plottable_data(
        self,
        input_data=None,
        data_file=None,
        georef_file=None,
        reduce_to_mask=True,
        get_coords=True,
        mode="label",
    ):
        """
        Extracts label and coordinate data from xArray or NWCSAF file.

        Parameters
        ----------
        input_data : xarray, optional
            Dataset containing label data. Needs to be set if 'data_file' is None.
        data_file : str, optional
            Filepath of NWCsAF type label file. Needs to be set if 'input_data' is None.
        georef_file : str, optional
            Filepath of georef file from which lat and lon values read.
            If None, georef file from project settings will be used.
        reduce_to_mask : bool, optional
            If True, data will be reduced to mask specified in the project settings.
        get_coords : bool, optional
            If True, coordinates will be returned with the label data.
        mode : str, optional
            If 'label' (default), label data will be extract, if 'proba' classifier certainty will
            be extracted (only for labels predicted by Random Forest classifer)
        """
        if input_data is None and data_file is None:
            raise ValueError("No input data given!")

        elif input_data is None:
            input_data = xr.open_dataset(data_file)
        if mode == "label":
            data = input_data[self.params["cloudtype_channel"]][0]
        elif mode == "proba":
            data = np.amax(input_data["label_probability"], axis=2)
        else:
            raise ValueError("Unsupported mode!")

        # shrink to area, transform to numpy
        indices = None
        if reduce_to_mask:
            if self.masked_indices is None:
                self.masked_indices = dh.set_indices_from_mask(self.params)
            indices = self.masked_indices
        else:
            indices = np.where(~np.isnan(data))

        out_data = np.empty(data.shape)
        out_data[:] = np.nan
        data = np.array(data)[indices[0], indices[1]]
        out_data[indices[0], indices[1]] = data
        if mode == "label":
            out_data = nwc.switch_nwcsaf_version(out_data, target_version="v2018")
            td.merge_labels(out_data, self.params["merge_list"])
        if not get_coords:
            return out_data
        else:
            if georef_file is None:
                georef_file = self.params["georef_file"]
            if georef_file is None:
                x = input_data.coords["lon"]
                y = input_data.coords["lat"]
            else:
                georef = xr.open_dataset(georef_file)
                x = georef.coords["lon"]
                y = georef.coords["lat"]

            return out_data, x, y

    def plot_coocurrence_matrix(
        self, label_file, truth_file, normalize=True, show=True
    ):
        """
        Plots coocurrence Matrix between predicted labels and ground truth.

        Parameters
        ----------
        label_file : str
            Filepath of NWSCAF type predicted label file.
        truth_file : TYPE
            Filepath of NWSCAF type ground truth label file.
        normalize : bool, optional
            If True, values will be normalized.
        show : bool, optional
            If True, plot will be displayed to the user.
        """
        label_data = self.get_plottable_data(
            data_file=label_file, reduce_to_mask=True, get_coords=False
        )
        truth_data = self.get_plottable_data(
            data_file=truth_file, reduce_to_mask=True, get_coords=False
        )
        conf.plot_coocurrence_matrix(
            label_data, truth_data, normalize=normalize, show=show
        )
