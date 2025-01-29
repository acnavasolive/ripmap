# ripmap

`ripmap` is a versatile and user-friendly toolbox to further analyze and curate SWR and IED events by reducing their dimensionality and visualizing them in a 2D space. 

**Summary**: The way `ripmap` works, is by first performing an analysis of the topological features of the data, by (i) computing the mean intrinsic dimension of the data in the original space, and (ii) characterizing the shape of the cloud using persistent homology. Then, data is reduced to a lower dimensional space and plotted in 2D using UMAP. `ripmap` GUI offers the possibility to perform the dimensionality reduction using different UMAP parameter values at the same time, and select the parameters that better segregate IED-detected from SWR-detected events. If SWR & IED events are clearly segregated, the user can visualize a clustering of the UMAP cloud by specifying “do_clusters = 1” in the GUI. Pressing “continue”, a new figure appears with an interactive plot that allows to select any custom area and display the events inside it. In this way, relating the position of an event in the UMAP cloud with its waveform becomes very intuitive. It can happen that the UMAP cloud shows no segregation and clustering is not possible, so an alternative curation approach can be taken by specifying “do_clusters=0” in the GUI. What happens then is that the 2D UMAP cloud is projected into a 1D axis and and binned in equidistant segments. The new figure shows the mean waveform of the events of each bin, which typically transitions from a SWR-like waveform to an IED-like waveform. Individual events can also be displayed here.



## System requirements

This toolbox does not need excessive computational requirements, and it has been tested on computers with both macOS (Sequoia Version 15.2, 32GB Apple M1 Pro) and Ubuntu (Ubuntu 22.04.3 LTS on a Lenovo ThinkPad L14 Gen 4 with 16 GB of memory, a 13th Gen Intel® Core™ i5-1335U × 12 processor and Mesa Intel® Graphics) operating systems.


## Installation guide

Tests were run on Python 3.11.2 with the following library versions: h5py==3.12.1, hdbscan==0.8.39, matplotlib==3.9.2, numba==0.60.0, numpy==2.0.2, permetrics==2.0.0, persim==0.3.7, ripser==0.6.10, scikit-learn==1.5.2, scipy==1.14.1, tqdm==4.67.0, umap-learn==0.5.7

If conda is installed on the system, an environment for installing all necessary libraries can be created as follows:
```
# Create the environment
conda create -n ripmap python=3.11.2 matplotlib==3.9.2 numpy==2.0.2 scikit-learn==1.5.2 scipy==1.14.1 

# Activate the environment
conda activate ripmap

# Install necessary packages
pip install h5py==3.12.1 hdbscan==0.8.39 numba==0.60.0 permetrics==2.0.0 persim==0.3.7 ripser==0.6.10 tqdm==4.67.0 umap-learn==0.5.7
 
# And clone this respository
git clone https://github.com/acnavasolive/ripmap.git
```


## Instructions for use & Demo

For this particular demo, patient number 20 has been included, and the original recording plus detected HFOs and IEDs times can be found at "ripmap/data/ER20_micro1.mat". This .mat file contains:
	- `lfp`: (timestamps x nchannels) matrix with raw LFP
	- `srate`: sampling rate in Hz (in this case, 2048Hz)
	- `detection_chan`: channel where SWR detection was made (in this case, channel 1)
	- `HFOs`: structure containing information about the HFOs detections. We will use the field `peaks`, a (#HFOs x 1) vector that contains the center of the events
	- `IEDs`: (#IEDs x 1) vector containing the center of the IED events


To initialize the toolbox, first open to the command terminal:
```
# navigate to the "ripmap" folder:
cd path-to-folder/ripmap

# activate the environment:
conda activate ripmap

# and run the main script:
python main.py
```


### Documentation


All the functions in `ripmap.py` are documented, and information can be accessed by typing `help(ripmap.fun)` in python. However, here we provide a small summary:

---

**```ripmap.event_curation(lfp, sf, t_swrs, t_ieds, id_fps=[], win_size_show=0.075, win_size_umap=0.075, do_detrend=True, do_zscore=False, list_n_neighbors=[10, 50, 100, 200], list_min_dists=[0.0, 0.1, 0.2, 0.3], intrinsic_dimension=4, n_elements=30, n_axis_bins=9, file_name='', saveas_folder='')```**

Implements an event curation based on waveform similarity, using dimensionality reduction. First, multiple embeddings for all the combinations of `n_neighbors` and `min_dist` from the optinal input variables `list_n_neighbors` and `list_min_dists` are computed with UMAP and presented in a figure, along with  extra summary plots of topological features of the original space (mean intrinsic dimension, computed  using ABID) and persistant homology for betti number = 0. In this plot, the user has to specify the optimal UMAP's `n_neighbors` and `min_dist` parameters, and if events can be divided into clusters or not. When the `Continue` button is pressed, the interactive curation GUI appears.
If there are clusters (`do_clusters=1`), the GUI will allow to select which is the cluster that represents the putative SWRs. Events that are not in the box will be labeled as `False` in the output variable `curated_labels`. If the `Finish` button is pressed, the GUI will close and the curated labels will be returned. If the `Update` button is pressed, then a new UMAP will be computed, and a new embedding will be shown. This process can be done multiple times until the cluster is as clean as possible.
If there are not clusters (`do_clusters=0`), then the whole UMAP cloud is divided into `n_axis_bins` bins along the main axis of the cloud shape. Events are projected into this axis, and the mean of the events of each bin are displayed with colors that reflect the amount of events that come from the SWR or IED detectors. The GUI contains two boxes to define from which to which bin (`min_bin` to `max_bin`) are the optimal events.

**Inputs**
- `lfp` (`np.ndarray`): array containing the LFP signal of the channel used to detect events
- `sf` (`float`): sampling frequency, in Hz
- `times` (`dict`): dictionary containing:
	- `swrs` (`np.ndarray): times (in sec) of the center of the events automatically detected by the SWR detector
	- `ieds` (`np.ndarray): times (in sec) of the center of the events automatically detected by the IED detector
	- `id_fps` (`np.ndarray`): indexes of the `t_swrs` variable that are False Positives. If not given, `id_fps = []`. UMAP will use all the entries that are in this dict. If there is no `ieds`, UMAP will be done only with `pSWRs`
- `power_spectrum` (`dict`): dictionary containing:
	- `swrs` (`np.ndarray`): power spectrum of all events detected by the SWR detector (same order)
	- `ieds` (`np.ndarray`): power spectrum of all events detected by the IED detector (same order)
- `use` (`list`): list of strings indicating what to use for UMAP. Options are: `lfp` and/or `power_spectrum` (`ps` or `powspctrm` are also accepted)
- `win_size_show` (`float`, *optional*): duration (in seconds) of the window before/after t_swrs (and t_ieds) over which to show events. By default, `win_size_show = 0.075`
- `win_size_umap` (`float`, *optional*): duration (in seconds) of the window before/after t_swrs (and t_ieds) over which to perform UMAP. By default, `win_size_umap = 0.075`
- `do_detrend` (`bool`, *optional*): to specify if dentrend should be applied to each event (`True`) or not (`False`). By default, `do_detrend = True`
- `do_zscore` (`bool`, *optional*): boolean to specify if zscore should be applied to each event (`True`) or not (`False`). By default, `do_zscore = False`
- `list_n_neighbors` (`np.ndarray` or `list`, *optional*): list of `n_neighbor` parameters to perform the Intrinsic Dimension and UMAP analysis with. By default, `list_n_neighbors = [10, 50, 100, 200]`
- `list_min_dists` (`np.ndarray` or `list`, *optional*): list of `min_dist` parameters to compute UMAP with. By default, `list_min_dists = [0.0, 0.1, 0.2, 0.3]`
- `intrinsic_dimension` (`int`, *optional*): integer to specify intrinsic dimension. It should be 4, but if the `Intrinsic dimension` plot shows something different, close and re-run this function changing this variable
- `n_elements` (`int`, *optional*): number of elements shown in the persistent homology analysis for betti number H=0. By default, `n_elements = 30`
- `n_axis_bins` (`int`, *optional*): number of bins by which to divide the UMAP cloud, in case there is no possibility of clustering. By default, `n_axis_bins = 9`
- `axis_method` (`string`, *optional*): method to use for computing the axis.
	- "centroids": computes the axis from the centroid of events from each detector, and traces a line between them (default). Automatically switches to "fit"` if no IED or no SWR data is provided
	- "fit": fits all data to a quadratic line (`topol_utils.fit_axis`). 
- `do_axis_grid` (`bool`, *optional*): in axis method: plot events over and below the axis in different subplots. By default, `do_axis_grid = False`
- `plot_fp_separately` (`bool`, *optional*): in axis method: plot False Positives separately. By default, `plot_fp_separately = False`
- `file_name` (`string`, *optional*): name of the file, to show it in plots (if provided)
- `saveas_folder` (`string`, *optional*): full path to folder in which to save plots (if provided)
- `save_format` (`string`, *optional*): format to save the figure. By default, `save_format='png'`

**Outputs**
- `curated_swrs` (`np.ndarray`): boolean array with the curated labels for `t_swrs`, selected through the interactive plots
- `curated_ieds` (`np.ndarray`): boolean array with the curated labels for `t_ieds`, selected through the interactive plots
- `events` (`np.ndarray`): array of size (#events, time) with all events to be curated
- `params` (`dict`): a dictionary with all the parameters

----

**```ripmap.plot_curated_events(events, curated_labels, params, from_swr_detector=None)```**

**Inputs**
- `events` (`np.ndarray`): array of size (#events, time) with all events to be curated
- `curated_labels` (`np.ndarray`): boolean array of size (#events,) indicating if the curation has classified each event as SWR (`True`) or IED (`False`).
- `params` (`dict`): dictionary of parameters, including: `id_fps`, `win_size`, `do_detrend`, `do_zscore`, `list_n_neighbors`, `list_min_dists`, `max_B`, `intrinsic_dimension`, `n_elements`, `n_axis_bins`, `file_name`, `saveas_folder`, `n_neighbors`, `min_dist`, `do_cluster`, 
	`embedding`
- `from_swr_detector` (`np.ndarray`): array of size (#events,) specifying for each event how was it detected:
	- 0 : from IED detector
	- 1 : from SWR detector
	- 2 : manually labeled as FP


**Returns**:
- `finish` (`boolean`): finish analysis or not

----

**```ripmap.manual_inspection(lfp, sf, t_curated, events_in_screen=50, file_name='', saveas_folder='')```**

**Inputs**
- `lfp` (`np.ndarray`): array containing the LFP signal of the channel used to detect events
- `sf` (`float`): sampling frequency, in Hz
- `t_swrs` (`np.ndarray`): array containing the times (in seconds) of the center of the events automatically detected by the SWR detector
- `t_ieds` (`np.ndarray`): array containing the times (in seconds) of the center of the events automatically detected by the IED detector
- `events_in_screen` (`int`, *optional*): number of events in screen. By default, 5x10
- `win_size` (`int`): length of the displayed ripples in miliseconds
- `file_name` (`string`, *optional*): name of the file, to show it in plots (if provided)
- `saveas_folder` (`string`, *optional*): full path to folder in which to save plots (if provided)

**Returns**

It always writes the curated events begin and end times in saveas_folder
- `curated_ids`: (events,) boolean array with `True` for events that have been selected, and `False` for events that had been discarded


----

### Demo pipeline

1. As described above, the first thing `ripmap` does is creating the events to be analysed, which are snippets around the center of the detected events. Snippets are preprocessed to optimise SWR-IED seggregation (`win_size_show = 0.100`; `win_size_umap = 0.020`; `do_detrend = True`; `do_zscore = False`; lines 28 to 31 in `main.py`; see Supplementary Figures 11 & 12). 

2. Then, different UMAP embeddings are created based on two parameters:
	- `n_neighbors`: this parameter controls how UMAP balances local versus global structure in the data. It does this by constraining the size of the local neighborhood UMAP will look at when attempting to learn the manifold structure of the data. This means that low values of n_neighbors will force UMAP to concentrate on very local structure (potentially to the detriment of the big picture), while large values will push UMAP to look at larger neighborhoods of each point when estimating the manifold structure of the data, losing fine detail structure for the sake of getting the broader of the data (see [documentation](https://umap-learn.readthedocs.io/en/latest/parameters.html#n-neighbors))
	- `min_dist`: this parameter controls how tightly UMAP is allowed to pack points together. It, quite literally, provides the minimum distance apart that points are allowed to be in the low dimensional representation. This means that low values of `min_dist` will result in clumpier embeddings. This can be useful if you are interested in clustering, or in finer topological structure. Larger values of `min_dist` will prevent UMAP from packing points together and will focus on the preservation of the broad topological structure instead (see [documentation](https://umap-learn.readthedocs.io/en/latest/parameters.html#min-dist))

	Because the UMAP cloud will have different shapes depending on these parameters, `ripmap` automatically explores the combination of several values (in particular, `list_n_neighbors = [10, 50, 100]`, and `list_min_dists = [0.0, 0.1, 0.2]`). Each dot represents an event, and is coloured based on their label: pIEDs (red: putative IEDs, coming from the IED detector), pSWRs (blue: putative SWRs, coming from the HFO detector), or FPs (grey: manual identification of false positives in the HFO list).  This is the part that takes the most time.

	<p align="center" width="100%">
		<img src="https://github.com/acnavasolive/ripmap/blob/main/figures/readme-01.png" width="800">
	</p>

    At the end of this process, the first screen is displayed, showing:
	- The intrinsic dimension of the high dimensional cloud, depending on the number of neighbors used. The green dotted line depicts the intrinsic dimension used for the analysis (in this case is 4D; see Figure 4).
	- Persistent homology analisys, barcodes reflect the “life span” of a connected component (Betti β0). The number of long barcodes represent the number of connected components in the original dimension cloud.
	- 3 x 3 UMAP plots, for all combinations of `list_n_neighbors` and `list_min_dists` values. 
	- 3 boxes to specify which UMAP cloud you want to explore and in which way.
		- `n_neighbors` and `min_dist`: which UMAP cloud you want to explore. For this demo, write `n_neighbors=50` and `min_dist=0.1`
		- `do_cluster`: 1 to make clusters, 0 to not make clusters. For this demo, write `do_cluster=1`.
  
   Press `Continue`.

3. Because we wrote `do_cluster=1`, the next screen shows the UMAP cloud, coloured with the same colors as previously (left), and coloured by an unsupervised clustering (right). The mean shape of the event is plotted over each cluster. If we click on the left panel, and press `Plot events`, we can visualise the events enclosed by the drawn shape.

    <p align="center" width="100%">
    	<img src="https://github.com/acnavasolive/ripmap/blob/main/figures/readme-02.png" width="800">
    </p>

    We can press `Esc` in our keyboard, draw another shape, and repeat the process.

    <p align="center" width="100%">
    	<img src="https://github.com/acnavasolive/ripmap/blob/main/figures/readme-03.png" width="800">
    </p>
    
    There are several options here:
	- `Back`: go back to the previous screen. It will not repeat the computations, so it’s immediate.
	- `Update`: creates a new UMAP from scratch based on the selected points.
	- `Project` to axis: the selected points are projected into a line, and divided into different segments. This is an alternative way to explore the data and allow fast manual curation.
	- `Finish`: take the selected points as final curated events, and finish the ongoing analysis.

	For this demo, select the middle cluster and click on `Project to axis`, but we encourage the reader to try the functionality of all of them.

4. The next screen shows only the selected events of the UMAP cloud, projected into an axis that goes from the centroid of the `pIEDs` to the centroid of the `pSWRs` (left, up). In gray, we can see that `FPs` lay in the middle part, as expected. On the bottom of the UMAP cloud, there is a histogram of the different event types across this drawn axis. The axis is divided into `n_axis_bins` bins, that by default, `is 9 (line 37 in main.py), but can be set into any number. `Each bin subpanel show the mean waveform of the events contained in that bin. We can again visualise the events that are contain into each bin by typing the bin number in the `min_bin` and `max_bin` boxes. This procedure allows again, a very fast visualisation and curation of the data.

    <p align="center" width="100%">
    	<img src="https://github.com/acnavasolive/ripmap/blob/main/figures/readme-04.png" width="800">
    </p>

    The axis can be also generated by fitting the data (press `Change axis to fit`). This could be set as the default mode if wanted (with `axis_method`, line 38 in main.py). There are other two additional options, set by `do_axis_grid` (line 39) and `fp_separately` (line 40), that separate events also depending in they are above/below the axis, and draw the FPs separately.

    <p align="center" width="100%">
    	<img src="https://github.com/acnavasolive/ripmap/blob/main/figures/readme-05.png" width="800">
    </p>

5. The last screen of this toolbox is a manual curation of the selected events, so that they can be validated manually, if desired. Here, the user has to click on the events to be discarded. Clicking on `Advance` changes the displayed events to the next 50 snippets, but the selection is always kept, so when `Go back` is clicked, the discarded events will again be shown in red. The `Discard all` and `Keep all` buttons reset the screen to discard or keep all, respectively. Finally, the `Save button` finishes the curation process, showing a last screen with all the curated events together, and saving the results in a .mat file in the `ripmap/results` folder, called "ER20_micro1.mat_curated_labels.mat". This file contains:
	- `curated_times`: (1 x #curated events), with the time of the events that were selected after ripmap curation.
	- `params`: dictionary with all the parameter values used for ripmap.
	- `events`: (#HFOs+#IEDs x snippet samples) matrix with the events used for ripmap
	- `selected_hfos`: (1 x #HFOs) boolean vector, with 1s in the HFO detections that had been selected using UMAP, and 0s for the events that were discarded.
	- `selected_ieds`: (1 x #IEDs) boolean vector, with 1s in the IED detections that had been selected using UMAP, and 0s for the events that were discarded.
	- `curated_manually`: (1 x #selected events) boolean vector, with 1s in the selected events, and 0s for the events that were discarded during manual curation.

<p align="center" width="100%">
	<img src="https://github.com/acnavasolive/ripmap/blob/main/figures/readme-06.png" width="800">
</p>

<p align="center" width="100%">
	<img src="https://github.com/acnavasolive/ripmap/blob/main/figures/readme-07.png" width="800">
</p>
