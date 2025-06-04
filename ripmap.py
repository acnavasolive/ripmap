# Imports
import os, sys
import numpy as np
# For interactive plots
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.widgets import TextBox, Button, PolygonSelector
from matplotlib.backend_bases import MouseButton
from matplotlib.path import Path
# Topological utils
from topol_utils import *
# Manual curation
from manual_utils import *
# For detrending/z-score
import scipy.signal, scipy.stats
# For UMAP
import umap
# For clustering
import hdbscan
import sklearn.cluster as cluster

# Global variables
COLORS = np.array([[.7,.2,.2], [.2,.6,.9], [.5,.5,.5]])
LABELS = ['pIEDs', 'pSWRs', 'FPs']
CMAP = ListedColormap(COLORS)


def event_curation(lfp, sf, times, power_spectrum=[], use=['lfp'],
					win_size_show=0.075, win_size_umap=0.075, do_detrend=True, do_zscore=False, 
					list_n_neighbors=[10, 50, 100, 200], list_min_dists=[0.0, 0.1, 0.2, 0.3], 
					intrinsic_dimension=4, n_elements=30, n_axis_bins=9, axis_method='centroids', 
					plot_fp_separately=False, do_axis_grid=False,
					file_name='', saveas_folder='', save_format='png'):
	'''
	event_curation(lfp, sf, t_swrs, t_ieds, id_fps=[], win_size_show=0.075, win_size_umap=0.075, 
					do_detrend=True, do_zscore=False, 
					list_n_neighbors=[10, 50, 100, 200], list_min_dists=[0.0, 0.1, 0.2, 0.3], 
					intrinsic_dimension=4, n_elements=30, n_axis_bins=9, file_name='', saveas_folder='')

	Implements an event curation based on waveform similarity, using dimensionality reduction. First, multiple
	embeddings for all the combinations of 'n_neighbors' and 'min_dist' from the optinal input variables 
	'list_n_neighbors' and 'list_min_dists' are computed with UMAP and presented in a figure, along with 
	extra summary plots of topological features of the original space (mean intrinsic dimension, computed 
	using ABID) and persistant homology for betti number = 0. In this plot, the user has to specify the optimal
	UMAP's 'n_neighbors' and 'min_dist' parameters, and if events can be divided into clusters or
	not. When the 'Continue' button is pressed, the interactive curation GUI appears.

	If there are clusters (do_clusters=1), the GUI will allow to select which is the cluster that represents the 
	putative SWRs. Events that are not in the box will be labeled as 'False' in the output variable 'curated_labels'. 
	If the 'Finish' button is pressed, the GUI will close and the curated labels will be returned. If the 'Update' 
	button is pressed, then a new UMAP will be computed, and a new embedding will be shown. This process can be done 
	multiple times until the cluster is as clean as possible.

	If there are not clusters (do_clusters=0), then the whole UMAP cloud is divided into 'n_axis_bins' bins along
	the main axis of the cloud shape. Events are projected into this axis, and the mean of the events of each bin
	are displayed with colors that reflect the amount of events that come from the SWR or IED detectors. The GUI
	contains two boxes to define from which to which bin ('min_bin' to 'max_bin') are the optimal events.

	Inputs:
	-------
		lfp (np.ndarray): 
			Array containing the LFP signal of the channel used to detect events
		sf (float):
			Sampling frequency, in Hz
		times (dict):
			Dictionary containing:
				'swrs': np.ndarray of times (sec) of the center of the events automatically detected by the SWR detector
				'ieds': np.ndarray of times (sec) of the center of the events automatically detected by the IED detector
				'id_fps': np.ndarray, containing the indexes of the 't_swrs' variable that are False Positives. 
						  If not given, id_fps = []
			UMAP will use all the entries that are in this dict. If there is no 'ieds', UMAP will be done only with pSWRs
		power_spectrum (dict):
			Dictionary containing:
				'swrs': np.ndarray of power spectrum of all events detected by the SWR detector (same order)
				'ieds': np.ndarray of power spectrum of all events detected by the IED detector (same order)
		use (list):
			List of strings indicating what to use for UMAP. Options are: 'lfp' and/or 'power_spectrum' ('ps' or 'powspctrm'
			are also accepted)
		win_size_show (float, optional):
			Duration (in seconds) of the window before/after t_swrs (and t_ieds) over which to show events
			By default win_size_show = 0.075
		win_size_umap (float, optional):
			Duration (in seconds) of the window before/after t_swrs (and t_ieds) over which to perform UMAP
			By default win_size_umap = 0.075
		do_detrend (bool, optional):
			To specify if dentrend should be applied to each event (True) or not (False).
			By default do_detrend = True
		do_zscore (bool, optional):
			Boolean to specify if zscore should be applied to each event (True) or not (False).
			By default do_zscore = False
		list_n_neighbors (np.ndarray or list, optional):
			List of 'n_neighbor' parameters to perform the Intrinsic Dimension and UMAP analysis with. 
			By default list_n_neighbors = [10, 50, 100, 200]
		list_min_dists (np.ndarray or list, optional):
			List of 'min_dist' parameters to compute UMAP with
			By default list_min_dists = [0.0, 0.1, 0.2, 0.3]
		intrinsic_dimension (int, optional):
			Integer to specify intrinsic dimension. It should be 4, but if the 'Intrinsic dimension' plot 
			shows something different, close and re-run this function changing this variable
		n_elements (int, optional):
			Number of elements shown in the persistent homology analysis for betti number H=0.
			By default n_elements = 30
		n_axis_bins (int, optional):
			Number of bins by which to divide the UMAP cloud, in case there is no possibility of clustering.
			By default n_axis_bins = 9
		axis_method (string, optional):
			Method to use for computing the axis.
				'centroids' - computes the axis from the centroid of events from each 
							  detector, and traces a line between them (default)
							  Automatically switches to 'fit' if no IED or no SWR data is provided
				'fit' - fits all data to a quadratic line (topol_utils.fit_axis). 
		do_axis_grid (bool, optional):
			In axis method: plot events over and below the axis in different subplots.
			By default do_axis_grid = False
		plot_fp_separately (bool, optional):
			In axis method: plot False Positives separately. 
			By default plot_fp_separately = False
		file_name (string, optional):
			Name of the file, to show it in plots (if provided)
		saveas_folder (string, optional):
			Full path to folder in which to save plots (if provided)
		save_format (string, optional):
			Format to save the figure. By default save_format='png'

	Outputs:
	--------
		curated_swrs (np.ndarray): 
			Boolean array with the curated labels for 't_swrs', selected through the interactive plots
		curated_ieds (np.ndarray): 
			Boolean array with the curated labels for 't_ieds', selected through the interactive plots
		events (np.ndarray):
			Array of size (#events, time) with all events to be curated
		params (dict):
			A dict with all the parameters
	'''


	# ========= PREPARE DATA ===========================================================================

	# Extract variables from inputs
	t_swrs = times['swrs'] if 'swrs' in times else np.empty((0))
	t_ieds = times['ieds'] if 'ieds' in times else np.empty((0))
	id_fps = times['id_fps'] if 'id_fps' in times else []

	# Variables
	max_B = 0
	# Convert them to arrays
	list_n_neighbors = np.array(list_n_neighbors)
	list_min_dists = np.array(list_min_dists)
	# Make dict
	params = {'sf':sf, 'id_fps':id_fps, 'use':use, 'win_size_show':win_size_show, 'win_size_umap':win_size_umap, 'do_detrend':do_detrend, 'do_zscore':do_zscore, 
			  'list_n_neighbors':list_n_neighbors, 'list_min_dists':list_min_dists, 'max_B':max_B,
			  'intrinsic_dimension':intrinsic_dimension, 'n_elements':n_elements, 
			  'n_axis_bins':n_axis_bins, 'axis_method':axis_method, 'do_axis_grid':do_axis_grid,
			  'plot_fp_separately':plot_fp_separately,'file_name':file_name, 
			  'saveas_folder':saveas_folder, 'save_format':save_format}

	# Join times from SWR and IED arrays
	t_all = np.append(t_swrs, t_ieds)
	from_swr_detector = np.append(np.ones_like(t_swrs), np.zeros_like(t_ieds)).astype(int)
	if len(t_swrs) > 0:
		from_swr_detector[id_fps] = 2
	# Update params
	params['t_all'] = t_all
	params['from_detector'] = from_swr_detector
	# Make matrix of events
	id_win = np.arange(-win_size_show*sf, win_size_show*sf +1).astype(int).reshape(1,-1)
	id_events = (t_all*sf).astype(int).reshape(-1,1)
	lfp_events = lfp[id_win+id_events]
	# Process data
	if do_detrend: lfp_events = scipy.signal.detrend(lfp_events, axis=1, type='linear')
	if do_zscore: lfp_events = scipy.stats.zscore(lfp_events, axis=1)

	# What to use for umap
	umap_events = np.empty((len(t_all), 0))
	if 'lfp' in use:
		# Make matrix of umap_events
		id_win = np.arange(-win_size_umap*sf, win_size_umap*sf +1).astype(int).reshape(1,-1)
		id_events = (t_all*sf).astype(int).reshape(-1,1)
		lfp_umap = lfp[id_win+id_events]
		# Process data
		if do_detrend: lfp_umap = scipy.signal.detrend(lfp_umap, axis=1, type='linear')
		if do_zscore: lfp_umap = scipy.stats.zscore(lfp_umap, axis=1)
		umap_events = np.hstack((umap_events, lfp_umap))
	if ('power_spectrum' in use) | ('ps' in use) | ('powspectrm' in use):
		if len(power_spectrum) > 0:
			# SWRs
			if ('swrs' in times) & ('swrs' in power_spectrum):
				ps_swrs = power_spectrum['swrs']
			elif ('swrs' in times) & ('swrs' not in power_spectrum):
				print("Power spectrum indicated to be used, but power_spectrum['swrs'] not provided")
				return None, None, None, None
			# IEDs
			if ('ieds' in times) & ('ieds' in power_spectrum):
				ps_ieds = power_spectrum['ieds']
			elif ('ieds' in times) & ('ieds' not in power_spectrum):
				print("Power spectrum indicated to be used, but power_spectrum['ieds'] not provided")
				return None, None, None, None
			# Append
			ps_all = np.vstack((ps_swrs, ps_ieds))
			umap_events = np.hstack((umap_events, ps_all))
			params['ps_freqs'] = power_spectrum['freqs']
		else:
			print('Power spectrum indicated to be used, but not provided')
			return None, None, None, None

	# ========= STEP 1: EVENT TOPOLOGY SUMMARY =========================================================

	# --- Intrinsic dimension ------

	# Comupte intrinsic dimension using ABID: Angle Based Intrinsic Dimensionality (by Erik Thordsen, Erich Schubert)
	abids = np.array([compute_abids(umap_events, n_neigh=n_neigh) for n_neigh in list_n_neighbors])
	# Remove nans
	if np.sum(np.isnan(abids)) > 0:
		abids = [intrdim[~np.isnan(intrdim)] for intrdim in abids]

	# --- Persistent homology ------

	# Compute diagrams
	diagrams, ds, thresh = compute_diagrams(umap_events, max_B=max_B)
	# Compute dense diagrams
	dense_diagrams, cumsum_distances = compute_dense_diagrams(diagrams, ds)

	# --- UMAP ------

	# Create embedding
	umap_embeddings = np.empty((len(list_n_neighbors), len(list_min_dists), len(umap_events), intrinsic_dimension))
	for iin, n_neigh in enumerate(list_n_neighbors):
		for iid, min_dist in enumerate(list_min_dists):

			embedding_umap = umap.UMAP(
					n_neighbors=n_neigh,
					min_dist=min_dist,
					n_components=intrinsic_dimension,
					metric='euclidean',
					metric_kwds=None,
					# random_state=42
					)
			print(f'making embedding (n_neighbors={n_neigh:.0f}, min_dist={min_dist:.1f})...\t\t\t\t', end='\r')
			# Fit the data
			embedding_umap.fit(umap_events)
			umap_embeddings[iin,iid] = embedding_umap.embedding_

	curated_labels = None
	do_finish = False
	while np.all(curated_labels == None):
		
		# ========= STEP 2: VISUALIZE =======================================================================
		
		# Make plot
		n_neighbors, min_dist, do_cluster = event_topology_summary(abids, diagrams, dense_diagrams, umap_embeddings, from_swr_detector, params)
		
		# Get selected embedding
		iin = np.argwhere(list_n_neighbors==n_neighbors)[0,0]
		iid = np.argwhere(list_min_dists==min_dist)[0,0]
		embedding = umap_embeddings[iin,iid]

		# Update parameters
		params['n_neighbors'] = int(n_neighbors)
		params['min_dist'] = min_dist
		params['do_cluster'] = do_cluster
		params['embedding'] = embedding


		# ========= STEP 3: CURATION =======================================================================

		if do_cluster: # Option 1: do_clusters=False -> Axis projection
			curated_labels, params = cluster_curation(embedding, lfp_events, umap_events, from_swr_detector, params)
			# Save info of events selected by axis
			params['selected_by_cluster'] = np.zeros((embedding.shape[0]))
			params['selected_by_cluster'] = curated_labels

		else: # Option 2: do_clusters=True -> Clusterization 
			curated_labels, params = axis_projection_curation(embedding, lfp_events, umap_events, from_swr_detector, params)
			# Save info of events selected by axis
			params['selected_by_axis'] = np.zeros((embedding.shape[0]))
			params['selected_by_axis'] = curated_labels

	# ========= STEP 4: RETURN =========================================================================

	# Outputs
	curated_swrs = curated_labels[from_swr_detector>0]
	curated_ieds = curated_labels[from_swr_detector==0]
	return curated_swrs, curated_ieds, umap_events, params


def event_topology_summary(abids, diagrams, dense_diagrams, umap_embeddings, from_swr_detector, params):
	'''
	event_topology_summary(abids, diagrams, dense_diagrams, umap_embeddings, from_swr_detector, params)
	
	Inputs:
	-------
		abids (np.ndarray):
			Angle Based Intrinsic Dimensionality array, which is the output topol_utils.compute_abids()
		diagrams (np.ndarray):
			Array with persistent homology element lifes, which is the output of tolo_utils.compute_diagrams()
		dense_diagrams (np.ndarray):
			Array with persistent homology element lifes, but computed as a function of the density,
			which is the output of tolo_utils.compute_dense_diagrams()
		umap_embeddings (np.ndarray):
			All the UMAP projections of 'umap_events' into the low-dimension embedding for each n_neighbors 
			and each min_dist
		from_swr_detector (np.ndarray):
			Array of size (#events,) specifying for each event how was it detected:
				0 - from IED detector
				1 - from SWR detector
				2 - manually labeled as FP
		params (dict):
			Dictionary of parameters, including: id_fps, win_size, do_detrend, do_zscore, 
			list_n_neighbors, list_min_dists, max_B, intrinsic_dimension, n_elements, 
			n_axis_bins, file_name, saveas_folder, n_neighbors, min_dist, do_cluster, 
			embedding
	
	Outputs:
	--------
		curated_labels (np.ndarray):
			Boolean array of size (#events,) indicating if the curation has classified
			each event as SWR (True) or IED (False).
	'''

	# Retrieve parameters
	list_n_neighbors = params['list_n_neighbors']
	list_min_dists = params['list_min_dists']
	max_B = params['max_B']
	intrinsic_dimension = params['intrinsic_dimension']
	n_elements = params['n_elements']
	file_name = params['file_name']
	saveas_folder = params['saveas_folder']
	save_format = params['save_format']
	use = params['use']

	# --- Generate the figure ------

	fig, axes = plt.subplots(1+len(list_n_neighbors), len(list_min_dists), figsize=(12,12))
	for iax in range(len(list_min_dists)-2):
		fig.delaxes(axes[0][iax+2])

	# --- Intrinsic dimension subplot ---

	mean_intrdim = np.mean(abids, axis=1)
	std_intrdim = np.std(abids, axis=1)
	# Plot distribution of mean+/-std of intrinsic dimensions
	axes[0,0].fill_between(list_n_neighbors, mean_intrdim-std_intrdim, mean_intrdim+std_intrdim, color='k', alpha=0.3, edgecolor=None)
	axes[0,0].plot(list_n_neighbors, mean_intrdim, color='k')
	axes[0,0].plot(list_n_neighbors, intrinsic_dimension*np.ones_like(list_n_neighbors), '--', color=[.1,.4,.2], label='current assumption')
	axes[0,0].set_xlabel('n_neighbor')
	axes[0,0].set_xticks(list_n_neighbors)
	axes[0,0].set_ylabel('intrinsic dimension')
	axes[0,0].set_ylim([0,10])
	axes[0,0].legend()
	axes[0,0].set_title('Intrinsic dimension analysis')

	# --- Plot diagrams plot for H=0 ---

	lifes = np.diff(diagrams[max_B][-n_elements:],axis=1).flatten()
	axes[0,1].barh(np.arange(n_elements), lifes, 0.8,left=diagrams[max_B][-n_elements:,0], color=[.3,.3,.3])
	# Plot dense plots
	lifes_dense = np.diff(dense_diagrams[max_B][-n_elements:],axis=1).flatten()
	for element in range(n_elements):
		axes[0,1].plot( [np.max(lifes)*.4, np.max(lifes)*.4 + lifes_dense[-(element+1)]/np.max(lifes_dense)*np.max(lifes)*.4], 
								  [n_elements/2.5*(2-element/n_elements)]*2, linewidth=.2, color=[.5,.5,.5])
	axes[0,1].text(np.max(lifes)*.4, n_elements/2.5-3, 'density', color=[.7,.7,.7])
	axes[0,1].set_xlabel('radius')
	axes[0,1].set_xticks([])
	axes[0,1].set_ylabel(r'$\beta_0$')
	axes[0,1].set_ylim([-1,n_elements])
	axes[0,1].set_yticks(np.arange(0,n_elements+1,10), labels=n_elements-np.arange(0,n_elements+1,10))
	axes[0,1].set_title('Persistent homology analysis')

	# --- UMAP subplots ---

	for iin, n_neigh in enumerate(list_n_neighbors):
		for iid, min_dist in enumerate(list_min_dists):
			# Sort: pIEDs - pSWRs - FPs
			idsort = np.argsort(from_swr_detector)
			# Plot cloud
			axes[1+iin,iid].scatter(umap_embeddings[iin,iid,idsort,0], umap_embeddings[iin,iid,idsort,1], c=from_swr_detector[idsort], s=5, alpha=.4, cmap=CMAP, vmin=0, vmax=2)
			axes[1+iin,iid].set_xticks([])
			axes[1+iin,iid].set_yticks([])
			if iin==0: axes[1+iin,iid].set_title(f'min_dist={min_dist:.1f}')
			if iid==0: axes[1+iin,iid].set_ylabel(f'n_neighbors={n_neigh:.0f}')

	# Legend
	fig.text(1-(len(list_min_dists)-2)/len(list_min_dists)*0.4, 1-1/len(list_n_neighbors)/8*4.3, LABELS[0], fontsize=14, color=COLORS[0])
	fig.text(1-(len(list_min_dists)-2)/len(list_min_dists)*0.4, 1-1/len(list_n_neighbors)/8*5, LABELS[1], fontsize=14, color=COLORS[1])
	fig.text(1-(len(list_min_dists)-2)/len(list_min_dists)*0.4, 1-1/len(list_n_neighbors)/8*5.7, LABELS[2], fontsize=14, color=COLORS[2])

	# --- Axes ---
	if len(file_name) > 0: plt.suptitle(f'{file_name} - events topology summary')
	else: plt.suptitle('Events topology summary')
	plt.tight_layout()

	# --- Buttons ---

	class InputValues:
		n_neighbors = None
		min_dist = None
		do_cluster = None
		textbox_n = None
		textbox_d = None
		textbox_c = None
		savefig = ''
		def continue_button(self, event):
			if len(self.textbox_n.text)>0:
				self.n_neighbors = float(self.textbox_n.text)
				print(f'input n_neighbors = {self.n_neighbors}')
			if len(self.textbox_d.text)>0:
				self.min_dist = float(self.textbox_d.text)
				print(f'input min_dist = {self.min_dist}')
			if len(self.textbox_c.text)>0:
				self.do_cluster = bool(float(self.textbox_c.text))
				print(f'input do_cluster = {self.do_cluster}')
			plt.savefig(self.savefig)
			plt.close()

	# Make callback class
	callback = InputValues()
	if len(file_name) > 0:
		callback.savefig = os.path.join(saveas_folder, f'{file_name}_events_topology_summary'+'_using'+''.join(use)+'.'+save_format)
	else:
		callback.savefig = os.path.join(saveas_folder, 'events_topology_summary'+'_using'+''.join(use)+'.'+save_format)
	# n_neighbors button
	axbox_n = plt.axes([1-(len(list_min_dists)-2)/len(list_min_dists)*0.7, 1-1/len(list_n_neighbors)/8*3, 0.1, 1/len(list_n_neighbors)/8]) if len(list_n_neighbors)>2 else plt.axes([0.95,0.95,0.05,0.05])
	callback.textbox_n = TextBox(axbox_n, 'n_neighbors: ')
	# min_dist button
	axbox_d = plt.axes([1-(len(list_min_dists)-2)/len(list_min_dists)*0.7, 1-1/len(list_n_neighbors)/8*4.5, 0.1, 1/len(list_n_neighbors)/8]) if len(list_n_neighbors)>2 else plt.axes([0.95,0.90,0.05,0.05])
	callback.textbox_d = TextBox(axbox_d, 'min_dist: ')
	# do_cluster button
	axbox_c = plt.axes([1-(len(list_min_dists)-2)/len(list_min_dists)*0.7, 1-1/len(list_n_neighbors)/8*6, 0.1, 1/len(list_n_neighbors)/8]) if len(list_n_neighbors)>2 else plt.axes([0.95,0.85,0.05,0.05])
	callback.textbox_c = TextBox(axbox_c, 'do_cluster: ')
	# Continue button
	axbut_continue = plt.axes([1-(len(list_min_dists)-2)/len(list_min_dists)*0.4, 1-1/len(list_n_neighbors)/8*3, 0.1, 1/len(list_n_neighbors)/8*1.5]) if len(list_n_neighbors)>2 else plt.axes([0.95,0.80,0.05,0.05])
	axbut = Button(axbut_continue, 'Continue')
	axbut.on_clicked(callback.continue_button)
	plt.show()

	# Extract input values
	n_neighbors = callback.n_neighbors
	min_dist = callback.min_dist
	do_cluster = callback.do_cluster

	return n_neighbors, min_dist, do_cluster


def make_axis_figure(fig, axes, from_swr_detector, params, embedding, lfp_events, umap_events, iteration=0):

	intrinsic_dimension = params['intrinsic_dimension']
	n_neighbors = params['n_neighbors']
	min_dist = params['min_dist']
	n_axis_bins = params['n_axis_bins']
	file_name = params['file_name']
	saveas_folder = params['saveas_folder']
	axis_method = params['axis_method']
	use = params['use']
	do_axis_grid = params['do_axis_grid']
	fpsep = params['plot_fp_separately']
	dops = ('power_spectrum' in use) | ('ps' in use) | ('powspectrm' in use)
	
	# Check axis method
	if (np.sum(from_swr_detector==0) == 0) | (np.sum(from_swr_detector==1) == 0):
		axis_method = 'fit'
		params['axis_method'] = axis_method

	# Get xs and ys
	xs = embedding[:,0] - np.mean(embedding[:,0])
	ys = embedding[:,1] - np.mean(embedding[:,1])
	# Fit a curve
	if axis_method == 'centroids':
		popt = centroid_curve(xs[from_swr_detector==1], ys[from_swr_detector==1], xs[from_swr_detector==0], ys[from_swr_detector==0])
	elif axis_method == 'fit':
		popt, _ = scipy.optimize.curve_fit(fit_axis, xs, ys)
	# Project to the curve
	xproj, yproj = project_to_curve(xs, ys, popt)
	# Fit a UMAP axis
	rini = np.array([xproj[np.argmin(xproj)], yproj[np.argmin(xproj)]])
	rend = np.array([xproj[np.argmax(xproj)], yproj[np.argmax(xproj)]])
	xdivs, ydivs = divide_axis(rini, rend, n_axis_bins, *popt)
	# Cluster events along the axis
	event_bins = bin_events_in_axis(xproj, xdivs)
	swr_iis_index_list = np.arange(np.max(event_bins))
	# Make histogram from xdivs and ydivs
	yhists = make_projected_histogram(xproj, xdivs, n_axis_bins, from_swr_detector)

	# Bin over/below the axis
	event_bins_isup = ((ys-yproj) > 0)

	# --- Plot UMAP cloud ---
	plt.subplot(3, int(n_axis_bins//1.5), (1,int(n_axis_bins//1.5)+1))
	for itype in range(len(COLORS)):
		plt.scatter(xs, ys, 6, color=COLORS[from_swr_detector], alpha=1, linewidth=0)
		# Plot projection
		# for i in np.argwhere(from_swr_detector>0).flatten():
		# 	plt.plot([xs[i],xproj[i]],[ys[i],yproj[i]], color=COLORS[from_swr_detector[i]], alpha=0.05)
	# Plot fit axis
	plt.plot(np.sort(xproj), fit_axis(np.sort(xproj), *popt), color='k', linewidth=0.8)
	# Plot division
	plt.scatter(xdivs, ydivs, 4, 'k')
	# Plot grid division
	if do_axis_grid:
		r = 100
		for xdiv, ydiv in zip(xdivs, ydivs):
			xt, yt = axis_tangent(xdiv, r, *popt)
			plt.plot([xdiv,xdiv-(yt-ydiv)], [ydiv,ydiv+(xt-xdiv)], 'k', linewidth=0.5, alpha=0.8)
			plt.plot([xdiv,xdiv+(yt-ydiv)], [ydiv,ydiv-(xt-xdiv)], 'k', linewidth=0.5, alpha=0.8)
	# Axis
	plt.xlim([np.min([xs,ys])-0.1, np.max([xs,ys])+0.1])
	plt.ylim([np.min([xs,ys])-0.1, np.max([xs,ys])+0.1])
	plt.xticks([])
	plt.yticks([])
	plt.xlabel('UMAP 1')
	plt.ylabel('UMAP 2')
	# Legend
	if np.sum((xs<np.mean(xs)) & (ys<np.mean(ys))) > np.sum((xs<np.mean(xs)) & (ys>np.mean(ys))):
		plt.text(np.min(xs), np.max(ys)-(np.max(ys)-np.min(ys))*0.10, LABELS[0], color=COLORS[0])
		plt.text(np.min(xs), np.max(ys)-(np.max(ys)-np.min(ys))*0.15, LABELS[1], color=COLORS[1])
		plt.text(np.min(xs), np.max(ys)-(np.max(ys)-np.min(ys))*0.20, LABELS[2], color=COLORS[2])
	else:
		plt.text(np.min(xs), np.min(ys)+(np.max(ys)-np.min(ys))*0.10, LABELS[0], color=COLORS[0])
		plt.text(np.min(xs), np.min(ys)+(np.max(ys)-np.min(ys))*0.15, LABELS[1], color=COLORS[1])
		plt.text(np.min(xs), np.min(ys)+(np.max(ys)-np.min(ys))*0.20, LABELS[2], color=COLORS[2])

	# --- Plot both histograms separately ---
	plt.subplot(3, int(n_axis_bins//1.5), 2*int(n_axis_bins//1.5)+1)
	types = np.unique(from_swr_detector)
	n_type = len(types)
	xhist = np.linspace(np.min(xproj), np.max(xproj), yhists.shape[1])
	dxhist = xhist[1]-xhist[0]
	for itype in range(n_type):
		plt.bar(xhist+dxhist/2., yhists[itype], width=dxhist, color=COLORS[types[itype]], alpha=0.5)
	if np.abs(np.sum(yhists)-n_type) < 0.05:
		plt.ylabel('Event distribution')
	else:
		plt.ylabel('# events')
	plt.xlabel('UMAP axis')
	plt.plot(xhist, xhist*0, '-ok')
	plt.xlim([xhist[0]-dxhist, xhist[-1]+dxhist])
	plt.xticks(xhist, labels=1+np.arange(len(xhist)))

	# --- Plot mean events ---
	ylims = [np.inf, -np.inf]
	ylims_ps = [np.inf, -np.inf]
	updown_label = ['up','down']
	for b in range(n_axis_bins):
		for updw in range(1+do_axis_grid):
			# If there is up/down grid
			if do_axis_grid:
				ids = (event_bins==b) & (from_swr_detector<(3-fpsep)) & (event_bins_isup==(1-updw))
			# If all is merged
			else:
				ids = (event_bins==b) & (from_swr_detector<(3-fpsep))
			if np.sum(ids)>0:
				# Compute mean event
				mean_event = np.mean(lfp_events[ids,:], axis=0)
				std_event = np.std(lfp_events[ids,:], axis=0)
				ylims[0] = np.nanmin([ylims[0], np.nanmin(mean_event-std_event)])
				ylims[1] = np.nanmax([ylims[1], np.nanmax(mean_event+std_event)])
				# Get mean color
				n_pswr = np.sum(from_swr_detector[ids]==1)
				n_pied = np.sum(from_swr_detector[ids]==0)
				n_fps = np.sum(from_swr_detector[ids]==2)
				color = (COLORS[0]*n_pied + COLORS[1]*n_pswr + COLORS[2]*n_fps)/(n_pswr+n_pied+n_fps) if (n_pswr+n_pied+n_fps) > 0 else np.array([.6,.6,.6])
				# Plot LFP
				axes[updw*(1+dops),2+b].fill_between(np.arange(len(mean_event)), (mean_event-std_event), (mean_event+std_event), color=color, alpha=0.3)
				axes[updw*(1+dops),2+b].plot(np.arange(len(mean_event)), mean_event, color=color*0.8)
				# Title
				axes[updw*(1+dops),2+b].set_title(f'Bin {b+1}')
				axes[updw*(1+dops),2+b].set_yticks([])
				axes[updw*(1+dops),2+b].set_xticks([])
				# Plot FPs separately
				if fpsep:
					if do_axis_grid: 
						ids_fp = (event_bins==b) & (from_swr_detector==2) & (event_bins_isup==(1-updw))
					else: 
						ids_fp = (event_bins==b) & (from_swr_detector==2)
					if np.sum(ids_fp) > 0:
						mean_fp = np.mean(lfp_events[ids_fp,:], axis=0)
						std_fp = np.std(lfp_events[ids_fp,:], axis=0)
						h = 2*np.max(np.mean(lfp_events,axis=0))
						ylims[1] = np.nanmax([ylims[1], h+np.nanmax(mean_fp+std_fp)])
						# Plot LFP
						axes[updw*(1+dops),2+b].fill_between(np.arange(len(mean_fp)), h+(mean_fp-std_fp), h+(mean_fp+std_fp), color=COLORS[2], alpha=0.3)
						axes[updw*(1+dops),2+b].plot(np.arange(len(mean_fp)), h+mean_fp, color=COLORS[2]*0.8)
				# Power spectrum
				if dops:
					# Get mean
					freqs = params['ps_freqs']
					mean_ps = np.mean(umap_events[ids,-len(freqs):], axis=0)
					std_ps = np.std(umap_events[ids,-len(freqs):], axis=0)
					ylims_ps[0] = np.nanmin([ylims_ps[0], np.nanmin(mean_ps-std_ps)])
					ylims_ps[1] = np.nanmax([ylims_ps[1], np.nanmax(mean_ps+std_ps)])
					axes[updw*(1+dops)+1,2+b].fill_between(freqs, (mean_ps-std_ps), (mean_ps+std_ps), color=color, alpha=0.3)
					axes[updw*(1+dops)+1,2+b].plot(freqs, mean_ps, color=color*0.8)
					axes[updw*(1+dops)+1,2+b].set_yticks([])
					if updw==do_axis_grid: # 0=up, 1=down
						axes[updw*(1+dops)+1,2+b].set_xlabel('Freq (Hz)')
					else:
						axes[updw*(1+dops)+1,2+b].set_xticks([])
					if fpsep & np.sum((event_bins==b) & (from_swr_detector==2))>0:
						mean_fp = np.mean(umap_events[ids_fp,-len(freqs)], axis=0)
						std_fp = np.std(umap_events[ids_fp,-len(freqs)], axis=0)
						h = 2*np.max(np.mean(umap_events[:,-len(freqs):],axis=0))
						ylims_ps[1] = np.nanmax([ylims_ps[1], h+np.nanmax(mean_fp+std_fp)])
						# Plot LFP
						axes[updw*(1+dops),2+b].fill_between(freqs, h+(mean_fp-std_fp), h+(mean_fp+std_fp), color=COLORS[2], alpha=0.3)
						axes[updw*(1+dops),2+b].plot(freqs, h+mean_fp, color=COLORS[2]*0.8)
			else:
				if updw==0:
					axes[updw*(1+dops),2+b].set_title(f'Bin {b+1}')
				for irow in range(axes.shape[0]):
					axes[irow,2+b].set_xticks([])
					axes[irow,2+b].set_yticks([])
			axes[updw*(1+dops),2].set_ylabel(updown_label[updw])
			if do_axis_grid & dops: axes[updw*(1+dops)+1,2].set_ylabel(updown_label[updw])

	# Plot mean events
	for b in range(n_axis_bins):
		for updw in range(1+do_axis_grid):
			axes[updw*(1+dops),2+b].set_ylim(ylims+np.array([-0.1,0.1]))
			if dops:
				axes[updw*(1+dops)+1,2+b].set_ylim(ylims_ps)

	# --- Axes ---
	if len(file_name) > 0:
		plt.suptitle(f'{file_name} - UMAP axis : {intrinsic_dimension}D, n_neighbors={n_neighbors:.0f}, min_dist={min_dist:.1f}, using {"+".join(use)}')
	else:
		plt.suptitle(f' UMAP axis - {intrinsic_dimension}D, n_neighbors={n_neighbors:.0f}, min_dist={min_dist:.1f}, using {"+".join(use)} (iteration {iteration})')

	return event_bins, event_bins_isup


def axis_projection_curation(embedding, lfp_events, umap_events, from_swr_detector, params):
	'''
	axis_projection_curation(embedding, lfp_events, umap_events, from_swr_detector, params)
	
	Inputs:
	-------
		embedding (np.ndarray):
			UMAP projection of 'lfp_events' into the low-dimension embedding
		lfp_events (np.ndarray):
			Array of size (#events, time) with all events to be curated
		umap_events (np.ndarray):
			Array of size (#events, time) with all inputs to umap
		from_swr_detector (np.ndarray):
			Array of size (#events,) specifying for each event how was it detected:
				0 - from IED detector
				1 - from SWR detector
				2 - manually labeled as FP
		params (dict):
			Dictionary of parameters, including: id_fps, win_size, do_detrend, do_zscore, 
			list_n_neighbors, list_min_dists, max_B, intrinsic_dimension, n_elements, 
			n_axis_bins, file_name, saveas_folder, n_neighbors, min_dist, do_cluster, 
			embedding
	
	Outputs:
	--------
		curated_labels (np.ndarray):
			Boolean array of size (#events,) indicating if the curation has classified
			each event as SWR (True) or IED (False).
	'''

	# Retrieve parameters
	intrinsic_dimension = params['intrinsic_dimension']
	n_neighbors = params['n_neighbors']
	min_dist = params['min_dist']
	n_axis_bins = params['n_axis_bins']
	file_name = params['file_name']
	saveas_folder = params['saveas_folder']
	axis_method = params['axis_method']
	save_format = params['save_format']
	use = params['use']
	do_axis_grid = params['do_axis_grid']
	plot_fp_separately = params['plot_fp_separately']

	# --- Generate the figure ------
	n_rows = 1
	if ('power_spectrum' in use) | ('ps' in use) | ('powspectrm' in use):
		n_rows +=1
	if do_axis_grid:
		n_rows *= 2
	fig, axes = plt.subplots(n_rows, n_axis_bins+2, figsize=((n_axis_bins+2)*1.5,4))
	axes = axes.reshape(n_rows, n_axis_bins+2)
	for irow in range(n_rows):
		fig.delaxes(axes[irow,0])
		fig.delaxes(axes[irow,1])
	event_bins, event_bins_isup = make_axis_figure(fig, axes, from_swr_detector, params, embedding, lfp_events, umap_events)

	# --- Buttons ---

	class InputValues:
		min_bin = None
		max_bin = None
		min_bin_up = None
		max_bin_up = None
		min_bin_down = None
		max_bin_down = None
		textbox_min = None
		textbox_max = None
		textbox_min_up = None
		textbox_max_up = None
		textbox_min_down = None
		textbox_max_down = None
		go_back = False
		savefig = ''
		lfp_events = None
		curated_labels = None
		params = None
		method_b = None
		event_bins = None
		event_bins_isup = None
		axes = None

		def back_button(self, event):
			self.go_back = True
			plt.close()

		def events_button(self, event):
			if self.params['do_axis_grid']:
				curated_labels_plot = np.copy(self.curated_labels)
				min_bin_up = float(self.textbox_min_up.text)-1 if len(self.textbox_min_up.text)>0 else 0
				min_bin_down = float(self.textbox_min_down.text)-1 if len(self.textbox_min_down.text)>0 else 0
				max_bin_up = float(self.textbox_max_up.text)-1 if len(self.textbox_max_up.text)>0 else np.inf
				max_bin_down = float(self.textbox_max_down.text)-1 if len(self.textbox_max_down.text)>0 else np.inf
				curated_labels_plot = ((event_bins>=min_bin_up) & (event_bins<=max_bin_up) & event_bins_isup) | ((event_bins>=min_bin_down) & (event_bins<=max_bin_down) & (~event_bins_isup))
			else:
				curated_labels_plot = np.copy(self.curated_labels)
				min_bin = float(self.textbox_min.text)-1 if len(self.textbox_min.text)>0 else 0
				max_bin = float(self.textbox_max.text)-1 if len(self.textbox_max.text)>0 else np.inf
				curated_labels_plot = (event_bins>=min_bin) & (event_bins<=max_bin)
			plot_curated_events(self.lfp_events, curated_labels_plot, self.params, from_swr_detector=from_swr_detector)

		def finish_button(self, event):
			if self.params['do_axis_grid']:
				if len(self.textbox_min_up.text)>0:
					self.min_bin_up = float(self.textbox_min_up.text)-1
					print(f'input min_bin_up = {self.min_bin_up+1}')
				if len(self.textbox_max_up.text)>0:
					self.max_bin_up = float(self.textbox_max_up.text)-1
					print(f'input max_bin_up = {self.max_bin_up+1}')
				if len(self.textbox_min_down.text)>0:
					self.min_bin_down = float(self.textbox_min_down.text)-1
					print(f'input min_bin_down = {self.min_bin_down+1}')
				if len(self.textbox_max_down.text)>0:
					self.max_bin_down = float(self.textbox_max_down.text)-1
					print(f'input max_bin_down = {self.max_bin_down+1}')
			else:
				if len(self.textbox_min.text)>0:
					self.min_bin = float(self.textbox_min.text)-1
					print(f'input min_bin = {self.min_bin+1}')
				if len(self.textbox_max.text)>0:
					self.max_bin = float(self.textbox_max.text)-1
					print(f'input max_bin = {self.max_bin+1}')
			plt.savefig(self.savefig)
			plt.close()

		def method_button(self, event):
			self.params['axis_method'] = 'fit' if (self.params['axis_method'] == 'centroids') else 'centroids'
			plt.subplot(3, int(n_axis_bins//1.5), (1,int(n_axis_bins//1.5)+1)); plt.cla()
			plt.subplot(3, int(n_axis_bins//1.5), 2*int(n_axis_bins//1.5)+1); plt.cla()
			for ii in range(n_axis_bins): 
				for irow in range(axes.shape[0]):
					axes[irow,2+ii].cla()
			self.method_b.label.set_text('Wait...')
			plt.draw() #redraw
			self.event_bins, self.event_bins_isup = make_axis_figure(fig, axes, from_swr_detector, params, embedding, lfp_events, umap_events)
			self.method_b.label.set_text('Change axis\nto '+('fit' if self.params['axis_method'] == 'centroids' else 'centroids'))
			plt.draw() #redraw

	# Make callback class
	callback = InputValues()
	callback.lfp_events = lfp_events
	callback.curated_labels = np.ones(embedding.shape[0]).astype(bool)
	callback.params = params
	callback.event_bins = event_bins
	callback.event_bins_isup = event_bins_isup
	callback.axes = axes
	savefig_details = 'umap_axis'+'_using'+''.join(use)+'_gridon'*do_axis_grid+'_FPseparately'*plot_fp_separately +'.'+save_format
	if len(file_name) > 0:
		callback.savefig = os.path.join(saveas_folder, f'{file_name}_{savefig_details}')
	else:
		callback.savefig = os.path.join(saveas_folder, savefig_details)
	# Min/Max buttons
	if do_axis_grid:
		# min/max up button
		axbox_min_up = plt.axes([0.97, 0.835, 0.025, 0.075])
		callback.textbox_min_up = TextBox(axbox_min_up, 'up min_bin: ', textalignment='center')
		axbox_max_up = plt.axes([0.97, 0.76, 0.025, 0.075])
		callback.textbox_max_up = TextBox(axbox_max_up, 'up max_bin: ', textalignment='center')
		# min/max down button
		axbox_min_down = plt.axes([0.97, 0.675, 0.025, 0.075])
		callback.textbox_min_down = TextBox(axbox_min_down, 'down min_bin: ', textalignment='center')
		axbox_max_down = plt.axes([0.97, 0.60, 0.025, 0.075])
		callback.textbox_max_down = TextBox(axbox_max_down, 'down max_bin: ', textalignment='center')
	else:
		# min_bin button
		axbox_min = plt.axes([0.95, 0.75, 0.025, 0.075])
		callback.textbox_min = TextBox(axbox_min, 'min_bin: ', textalignment='center')
		# max_bin button
		axbox_max = plt.axes([0.95, 0.65, 0.025, 0.075])
		callback.textbox_max = TextBox(axbox_max, 'max_bin: ', textalignment='center')
	# Back button
	axbut_back = plt.axes([0.91, 0.50, 0.085, 0.09])
	axbut_b = Button(axbut_back, 'Back')
	axbut_b.on_clicked(callback.back_button)
	# Plot events button
	axbut_events = plt.axes([0.91, 0.40, 0.085, 0.09])
	axbut_e = Button(axbut_events, 'Plot events')
	axbut_e.on_clicked(callback.events_button)
	# Finish button
	axbut_finish = plt.axes([0.91, 0.30, 0.085, 0.09])
	axbut_f = Button(axbut_finish, 'Finish')
	axbut_f.on_clicked(callback.finish_button)
	# Method button
	axbut_method = plt.axes([0.91, 0.16, 0.085, 0.13])
	axbut_m = Button(axbut_method, 'Change axis\nto '+('fit' if axis_method == 'centroids' else 'centroids'))
	axbut_m.on_clicked(callback.method_button)
	callback.method_b = axbut_m
	plt.show()

	# Extract input values
	if callback.go_back:
		# Go back
		curated_labels = None 
	else:
		# Return curated vector
		if do_axis_grid:
			curated_up = (callback.event_bins>=callback.min_bin_up) & (callback.event_bins<=callback.max_bin_up) & callback.event_bins_isup
			curated_down = (callback.event_bins>=callback.min_bin_down) & (callback.event_bins<=callback.max_bin_down) & (~callback.event_bins_isup)
			curated_labels = curated_up | curated_down
		else:
			min_bin = callback.min_bin
			max_bin = callback.max_bin
			curated_labels = (callback.event_bins>=min_bin) & (callback.event_bins<=max_bin)
		params['axis_bins'] = callback.event_bins

	return curated_labels, params


def cluster_curation(embedding, lfp_events, umap_events, from_swr_detector, params):
	'''
	cluster_curation(embedding, lfp_events, umap_events, from_swr_detector, params)
	
	Inputs:
	-------
		embedding (np.ndarray):
			UMAP projection of 'lfp_events' into the low-dimension embedding
		lfp_events (np.ndarray):
			Array of size (#events, time) with all events to be curated
		umap_events (np.ndarray):
			Array of size (#events, time) with all inputs to umap
		from_swr_detector (np.ndarray):
			Array of size (#events,) specifying for each event how was it detected:
				0 - from IED detector
				1 - from SWR detector
				2 - manually labeled as FP
		params (dict):
			Dictionary of parameters, including: id_fps, win_size, do_detrend, do_zscore, 
			list_n_neighbors, list_min_dists, max_B, intrinsic_dimension, n_elements, 
			n_axis_bins, file_name, saveas_folder, n_neighbors, min_dist, do_cluster, 
			embedding
	
	Outputs:
	--------
		curated_labels (np.ndarray):
			Boolean array of size (#events,) indicating if the curation has classified
			each event as SWR (True) or IED (False).
	'''

	# Retrieve parameters
	intrinsic_dimension = params['intrinsic_dimension']
	n_neighbors = params['n_neighbors']
	min_dist = params['min_dist']
	n_axis_bins = params['n_axis_bins']
	file_name = params['file_name']
	saveas_folder = params['saveas_folder']
	save_format = params['save_format']
	use = params['use']

	# Select the embedding
	curated_labels = np.ones_like(from_swr_detector).astype(bool)
	embedding_original = np.copy(embedding)

	# Prepare the iterative process
	iteration = 1
	do_update = True

	while do_update:

		if iteration > 1:
			# Create embedding
			embedding_umap = umap.UMAP(
					n_neighbors=int(n_neighbors),
					min_dist=min_dist,
					n_components=intrinsic_dimension,
					metric='euclidean',
					metric_kwds=None,
					# random_state=42
					)
			print(f'making embedding (n_neighbors={n_neighbors:.0f}, min_dist={min_dist:.1f})...\t\t\t\t', end='\r')
			# Fit the data
			embedding_umap.fit(lfp_events[curated_labels])
			embedding = embedding_umap.embedding_
			params['embedding'] = embedding

		# Take 1st and 2nd dimension
		xs = embedding[:,0] - np.mean(embedding[:,0])
		ys = embedding[:,1] - np.mean(embedding[:,1])

		# Cluster parameters depending on points in embedding
		min_clus_size = int(np.ceil(len(embedding)*0.1))
		min_samples = int(np.ceil(min_clus_size*0.05))

		# Cluster
		clusters = hdbscan.HDBSCAN(
				min_samples=min_samples,
				min_cluster_size=min_clus_size,
				allow_single_cluster = False,
			).fit_predict(embedding)
		n_clusters = np.max(clusters)+1

		# Make: pIEDs = cluster 0, pSWRs = cluster 1
		event_sort = np.argsort([np.mean(from_swr_detector[curated_labels][clusters==ci]) for ci in np.unique(clusters)])
		clusters_tmp = np.copy(clusters)
		for iclu in range(n_clusters):
			clusters[clusters_tmp==event_sort[iclu]] = iclu

		# --- Generate the figure ------

		fig, axes = plt.subplots(1,2, figsize=(14,7), sharey=True)

		# --- Plot UMAP cloud ---

		# Plot
		axes[0].scatter(xs, ys, 6, color=COLORS[from_swr_detector[curated_labels]], alpha=1, linewidth=0)
		axes[0].set_xticks([])
		axes[0].set_yticks([])
		axes[0].set_xlabel('UMAP 1')
		axes[0].set_ylabel('UMAP 2')
		axes[0].set_title('Colored by detector')
		# Legend
		if np.sum((xs<np.mean(xs)) & (ys<np.mean(ys))) > np.sum((xs<np.mean(xs)) & (ys>np.mean(ys))):
			axes[0].text(np.min(xs), np.max(ys)-(np.max(ys)-np.min(ys))*0.10, LABELS[0], color=COLORS[0])
			axes[0].text(np.min(xs), np.max(ys)-(np.max(ys)-np.min(ys))*0.15, LABELS[1], color=COLORS[1])
			axes[0].text(np.min(xs), np.max(ys)-(np.max(ys)-np.min(ys))*0.20, LABELS[2], color=COLORS[2])
		else:
			axes[0].text(np.min(xs), np.min(ys)+(np.max(ys)-np.min(ys))*0.10, LABELS[0], color=COLORS[0])
			axes[0].text(np.min(xs), np.min(ys)+(np.max(ys)-np.min(ys))*0.15, LABELS[1], color=COLORS[1])
			axes[0].text(np.min(xs), np.min(ys)+(np.max(ys)-np.min(ys))*0.20, LABELS[2], color=COLORS[2])

		# --- Cluster UMAP events ---

		# Plot UMAP clusters
		for i in np.unique(clusters):
			if i == -1: axes[1].scatter(xs[clusters==i], ys[clusters==i], s=6, color=[.8,.8,.8], edgecolor='none')
			else: axes[1].scatter(xs[clusters==i], ys[clusters==i], c=f'C{i+(i>=0)+(i>=2)}', s=6, edgecolor='none')
		axes[1].set_xticks([])
		axes[1].set_yticks([])
		axes[1].set_xlabel('UMAP 1')
		axes[1].set_ylabel('UMAP 2')
		axes[1].set_title('Draw a polygon with clicks\nto select cluster')
		for i in np.unique(clusters[clusters>=0]):
			dx = np.mean(xs[clusters==i])
			dy = np.mean(ys[clusters==i])
			lx = np.max(xs)-np.min(xs)
			ly = np.max(ys)-np.min(ys)
			# Plot mean events
			mean_event = np.mean(lfp_events[curated_labels][clusters==i], axis=0)
			std_event = np.std(lfp_events[curated_labels][clusters==i], axis=0)
			std_event = (std_event-np.min(std_event))/(np.max(np.mean(lfp_events[curated_labels],axis=0))-np.min(np.mean(lfp_events[curated_labels],axis=0)))
			mean_event = (mean_event-np.min(mean_event))/(np.max(np.mean(lfp_events[curated_labels],axis=0))-np.min(np.mean(lfp_events[curated_labels],axis=0)))
			axes[1].fill_between(dx-lx/6 + np.linspace(0,lx/3,lfp_events[curated_labels].shape[1]),
				dy-ly/6 + (mean_event-std_event)*ly/3, dy-ly/6 + (mean_event+std_event)*ly/3,
				color='k', alpha=0.4, edgecolor='none')
			axes[1].plot(dx-lx/6 + np.linspace(0,lx/3,lfp_events[curated_labels].shape[1]), dy-ly/6 + mean_event*ly/3, 'k', linewidth=1.2)

		# --- Axis ---
		if len(file_name) > 0:
			plt.suptitle(f'{file_name} - UMAP clustering (iteration {iteration})\n{intrinsic_dimension}D, n_neighbors={n_neighbors:.0f}, min_dist={min_dist:.1f}, using {"+".join(use)}')
		else:
			plt.suptitle(f'UMAP clustering - {intrinsic_dimension}D, n_neighbors={n_neighbors:.0f}, min_dist={min_dist:.1f}, using {"+".join(use)} (iteration {iteration})')

		# --- Buttons

		class InputPolygon:
			verts = None
			go_back = False
			do_axis = False
			do_finish = False
			iteration = 0
			savefig = ''
			lfp_events = None
			curated_labels = None
			params = None
			X = None
			Y = None
			embedding = None

			def line_select(self, verts):
				self.verts = verts

			def back_button(self, event):
				self.go_back = True
				plt.close()

			def update_button(self, event):
				self.iteration += 1
				plt.savefig(self.savefig)
				plt.close()

			def axis_button(self, event):
				self.do_axis = True
				plt.savefig(self.savefig)
				plt.close()

			def events_button(self, event):
				curated_labels_plot = np.copy(self.curated_labels)
				ids_in_embedding = np.argwhere(curated_labels_plot).flatten()
				# Events in polygon
				path = Path(self.verts)
				in_polygon = path.contains_points(np.hstack((self.X, self.Y)))
				curated_labels_plot[ids_in_embedding] = in_polygon
				print(f'self.lfp_events[ids_in_embedding].shape: {self.lfp_events.shape}')
				print(f'curated_labels_plot[ids_in_embedding].shape: {curated_labels_plot.shape}')
				print(f'from_swr_detector[ids_in_embedding].shape: {from_swr_detector.shape}')
				plot_curated_events(self.lfp_events[ids_in_embedding,:], curated_labels_plot[ids_in_embedding], self.params, from_swr_detector=from_swr_detector[ids_in_embedding])

			def plot_3d(self, event):
				fig = plt.figure(figsize=(10,10))
				axs = []
				for iplot, (dim1, dim2, dim3) in enumerate([(0,1,2), (0,1,3), (0,2,3), (1,2,3)]):
					axs.append( fig.add_subplot(2,2,iplot+1, projection='3d') )
					for i in np.unique(clusters):
						ids = (self.clusters==i)
						if i == -1: axs[iplot].scatter(self.embedding[ids,dim1], self.embedding[ids,dim2], self.embedding[ids,dim3], s=6, color=[.8,.8,.8], edgecolor='none')
						else: axs[iplot].scatter(self.embedding[ids,dim1], self.embedding[ids,dim2], self.embedding[ids,dim3], c=f'C{i+(i>=0)+(i>=2)}', s=6, edgecolor='none')
					axs[iplot].set_xticks([])
					axs[iplot].set_yticks([])
					axs[iplot].set_zticks([])
					axs[iplot].set_xlabel(f'UMAP {dim1+1}')
					axs[iplot].set_ylabel(f'UMAP {dim2+1}')
					axs[iplot].set_zlabel(f'UMAP {dim3+1}')
				plt.show()

			def finish_button(self, event):
				self.do_finish = True
				plt.savefig(self.savefig)
				plt.close()

		callback = InputPolygon()
		callback.iteration = iteration
		callback.lfp_events = lfp_events
		callback.curated_labels = np.copy(curated_labels)
		callback.params = params
		callback.X = xs.reshape(-1,1)
		callback.Y = ys.reshape(-1,1)
		print(embedding.shape)
		callback.embedding = embedding.copy()
		callback.clusters = clusters.copy()
		if len(file_name) > 0:
			callback.savefig = os.path.join(saveas_folder, f'{file_name}_cluster_iteration{iteration}.{save_format}')
		else:
			callback.savefig = os.path.join(saveas_folder, f'cluster_iteration{iteration}.{save_format}')
		# Back button
		axbut_back = plt.axes([0.91, 0.80, 0.08, 0.075])
		axbut_b = Button(axbut_back, 'Back')
		axbut_b.on_clicked(callback.back_button)
		# Update button
		axbut_update = plt.axes([0.91, 0.70, 0.08, 0.075])
		axbut_u = Button(axbut_update, 'Update')
		axbut_u.on_clicked(callback.update_button)
		# Do axis button
		axbut_axis = plt.axes([0.91, 0.60, 0.08, 0.075])
		axbut_a = Button(axbut_axis, 'Project to axis')
		axbut_a.on_clicked(callback.axis_button)
		# Plot events button
		axbut_events = plt.axes([0.91, 0.50, 0.08, 0.075])
		axbut_e = Button(axbut_events, 'Plot events')
		axbut_e.on_clicked(callback.events_button)
		# Finish button
		axbut_3d = plt.axes([0.91, 0.40, 0.08, 0.075])
		axbut_3 = Button(axbut_3d, 'Plot in 3D')
		axbut_3.on_clicked(callback.plot_3d)
		# Finish button
		axbut_finish = plt.axes([0.91, 0.30, 0.08, 0.075])
		axbut_f = Button(axbut_finish, 'Finish')
		axbut_f.on_clicked(callback.finish_button)
		# Interactive rectangle 
		print("\n	  click to select the correct cluster -->  release")
		RS = PolygonSelector(axes[1], callback.line_select,
												useblit=True,
												props=dict(color='k', linestyle='-', linewidth=2, fillstyle='none'),
												# draw_bounding_box=True,
												)
		print("Click on the figure to create a polygon.")
		print("Press the 'esc' key to start a new polygon.")
		plt.show()

		# --- Update ---

		# Update curation
		if np.all(callback.verts != None):
			print('Update curation')
			ids_in_embedding = np.argwhere(curated_labels).flatten()
			# Events in polygon
			path = Path(callback.verts)
			in_polygon = path.contains_points(np.hstack((callback.X, callback.Y)))
			curated_labels[ids_in_embedding] = in_polygon

		# Update variables for the while loop
		do_update = iteration < callback.iteration
		iteration = callback.iteration

	# If 'project to axis' button is pressed, then do axis analysis
	if callback.do_axis:
		# breakpoint()
		ids_in_embedding = np.argwhere(curated_labels).flatten()
		axis_curation, params = axis_projection_curation(embedding_original[curated_labels,:], lfp_events[curated_labels], umap_events[curated_labels], from_swr_detector[curated_labels], params)
		# Save info of events selected by axis
		params['selected_by_axis'] = np.zeros_like(curated_labels)
		params['selected_by_axis'][ids_in_embedding] = axis_curation
		# Add '-1' to params['axis_bins'] for events that were not in the axis projection
		bins_ = -1 * np.ones_like(curated_labels)
		bins_[ids_in_embedding] = 1+params['axis_bins']
		params['axis_bins'] = bins_
		if np.all(axis_curation == None):
			curated_labels = None
		else:
			curated_labels[ids_in_embedding] = axis_curation

	# Go back
	elif callback.go_back:
		curated_labels = None

	# Save info of events selected by axis
	params['selected_by_cluster'] = np.zeros((embedding.shape[0]))
	params['selected_by_cluster'] = curated_labels

	return curated_labels, params


def plot_curated_events(events, curated_labels, params, from_swr_detector=None):
	'''
	plot_curated_events(events, curated_labels, params, from_swr_detector=None)
	
	Inputs:
	-------
		events (np.ndarray):
			Array of size (#events, time) with all events to be curated
		curated_labels (np.ndarray):
			Boolean array of size (#events,) indicating if the curation has classified
			each event as SWR (True) or IED (False).
		params (dict):
			Dictionary of parameters, including: id_fps, win_size, do_detrend, do_zscore, 
			list_n_neighbors, list_min_dists, max_B, intrinsic_dimension, n_elements, 
			n_axis_bins, file_name, saveas_folder, n_neighbors, min_dist, do_cluster, 
			embedding
		from_swr_detector (np.ndarray):
			Array of size (#events,) specifying for each event how was it detected:
				0 - from IED detector
				1 - from SWR detector
				2 - manually labeled as FP

	Outputs:
	--------
		finish (boolean):
			Finish analysis or not
	'''

	# Retrieve parameters
	intrinsic_dimension = params['intrinsic_dimension']
	n_neighbors = params['n_neighbors']
	min_dist = params['min_dist']
	n_axis_bins = params['n_axis_bins']
	file_name = params['file_name']
	saveas_folder = params['saveas_folder']
	save_format = params['save_format']
	use = params['use']
	win_size_show = params['win_size_show']
	sf = params['sf']
	manual_inspection = params['manual_inspection'] if 'manual_inspection' in params else False
	text = 'manual_inspection' if manual_inspection else 'curated_events'

	
	# Function to normalize data
	def normalize_data(data):
		return (data - np.min(data)) / (np.max(data) - np.min(data))

	# Function to plot all events one next to another
	def plot_all_events(ax, events, curated_labels, sf=sf, wsize=win_size_show, normalize=False):
		if np.sum(curated_labels)>0:

			# Curated events
			curated_events = events[curated_labels]
			kmax = 1.5 if normalize else np.max(np.abs(curated_events))

			# Plot curated events
			n_cols = int(np.sqrt(len(curated_events)))
			dx, dy = 0, 0
			for ii,event in enumerate(curated_events):
				event = normalize_data(event) if normalize else event
				if np.all(from_swr_detector == None):
					ax.plot(dx+np.linspace(.05,.95,len(event)), 
							dy+event/kmax, linewidth=0.7)
				else:
					ax.plot(dx+np.linspace(.05,.95,len(event)), 
							dy+event/kmax, linewidth=0.7, color=COLORS[from_swr_detector[curated_labels][ii]])
				# Plot time scale
				if ii == 0:
					ax.plot([dx+.05,dx+.50], [dy+1.2, dy+1.2], linewidth=1, color='k')
					ax.text(dx+.05+(.50-.05)/2., dy+1.25, f'{wsize*1000:.0f}ms', size='small', horizontalalignment='center', verticalalignment='baseline')
				# # Plot V scale
				# if (ii == 0) & (not normalize):
				# 	ax.plot([dx+.05,dx+.05], [dy+1.2, dy+1], linewidth=1, color='k')
				# 	ax.text(dx+.20, dy+1.05, f'{wsize*1000:.0f}ms', size='small')
				# Prepare for next
				dx = dx+1
				if dx >= n_cols:
					dx = 0
					dy = dy-1
		# Draw axis
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_axis_off()
	
	# Class to Normalize events
	class NormalizeButton:
		do_normalize = False
		method_b = None
		ax = None
		def method_button(self, event):
			self.do_normalize = not self.do_normalize
			self.ax.cla()
			# Plot
			plot_all_events(self.ax, events, curated_labels, normalize=self.do_normalize)
			# Change name in button label
			self.method_b.label.set_text('See events\n'+('in mV' if self.do_normalize else 'normalized'))
			# Title and save
			if len(file_name) > 0:
				ax.set_title(f"{file_name} - all {text.replace('_',' ')}\n{intrinsic_dimension}D, n_neighbors={n_neighbors:.0f}, min_dist={min_dist:.1f}, using {'+'.join(use)}")
			else:
				plt.suptitle(f"All {text.replace('_',' ')} - {intrinsic_dimension}D, n_neighbors={n_neighbors:.0f}, min_dist={min_dist:.1f}, using {'+'.join(use)} (iteration {iteration})")
			plt.tight_layout()
			# Save
			if len(file_name) > 0:
				fig.savefig(os.path.join(saveas_folder, f'{file_name}_{text}.{save_format}'))
			else:
				fig.savefig(os.path.join(saveas_folder, f'{text}.{save_format}'))
			# redraw
			plt.draw()


	# Make figure
	fig, ax = plt.subplots(figsize=(16,12))
	plot_all_events(ax, events, curated_labels, normalize=False)

	# Method button
	callback = NormalizeButton()
	axbut_norm = plt.axes([0.88, 0.94, 0.07, 0.05])
	axbut_n = Button(axbut_norm, 'See events\n normalized')
	axbut_n.on_clicked(callback.method_button)
	callback.method_b = axbut_n
	callback.ax = ax

	# Title and save
	if len(file_name) > 0:
		ax.set_title(f"{file_name} - all {text.replace('_',' ')}\n{intrinsic_dimension}D, n_neighbors={n_neighbors:.0f}, min_dist={min_dist:.1f}, using {'+'.join(use)}")
	else:
		plt.suptitle(f"All {text.replace('_',' ')} - {intrinsic_dimension}D, n_neighbors={n_neighbors:.0f}, min_dist={min_dist:.1f}, using {'+'.join(use)} (iteration {iteration})")
	plt.tight_layout()
	if len(file_name) > 0:
		fig.savefig(os.path.join(saveas_folder, f'{file_name}_{text}.{save_format}'))
	else:
		fig.savefig(os.path.join(saveas_folder, f'{text}.{save_format}'))
	plt.show()


def manual_inspection(lfp, sf, t_swrs, t_ieds, params, events_in_screen=50, win_size=100, file_name='', saveas_folder=''):
	'''
	manual_inspection(lfp, sf, t_curated, events_in_screen=50, file_name='', saveas_folder='')
	
	Inputs:
	-------
		lfp (np.ndarray): 
			Array containing the LFP signal of the channel used to detect events
		sf (float):
			Sampling frequency, in Hz
		t_swrs (np.ndarray):
			Array containing the times (in seconds) of the center of the events 
			automatically detected by the SWR detector
		t_ieds (np.ndarray):
			Array containing the times (in seconds) of the center of the events 
			automatically detected by the IED detector
		events_in_screen (int, optional):
			Number of events in screen. By default, 5x10
		win_size (int):
			Length of the displayed ripples in miliseconds
		file_name (string, optional):
			Name of the file, to show it in plots (if provided)
		saveas_folder (string, optional):
			Full path to folder in which to save plots (if provided)

	Outputs:
	-------
		It always writes the curated events begin and end times in saveas_folder
		curated_ids: (events,) boolean array with 'True' for events that have been
			selected, and 'False' for events that had been discarded

	'''

	# Join times from SWR and IED arrays
	id_all = np.round(np.append(t_swrs, t_ieds)*sf).astype(int)
	from_swr_detector = np.append(np.ones_like(t_swrs), np.zeros_like(t_ieds)).astype(int)
	id_sort = np.argsort(id_all)
	id_all = id_all[id_sort]
	from_swr_detector = from_swr_detector[id_sort]

	plt.rc('xtick', labelsize=7.5)
	plt.rc('ytick', labelsize=7.5)
	plt.rcParams['axes.spines.left'] = False
	plt.rcParams['axes.spines.right'] = False
	plt.rcParams['axes.spines.top'] = False
	plt.rcParams['axes.spines.bottom'] = False
	timesteps = int((win_size*sf/1000)//2) # Timesteps to be shown
	oIn = saved_intervals(id_all, None)

	# Create figure
	n_suby = np.ceil(np.sqrt(events_in_screen)).astype(int)
	tries = np.hstack((np.arange(1,n_suby-1).reshape(-1,1), -np.arange(1,n_suby-1).reshape(-1,1))).flatten()
	k = 0
	while (events_in_screen%n_suby) != 0:
		n_suby = np.ceil(np.sqrt(events_in_screen)).astype(int)+tries[k]
		k += 1
	fig, axes = plt.subplots(n_suby,int(events_in_screen/n_suby), figsize=(15,10))
	fig.suptitle(f"Displaying ripples {oIn.index} to {oIn.n_og_len if (oIn.check_index(events_in_screen)) else oIn.index+events_in_screen } out of {oIn.n_og_len}",x=0.475)
	# Color definition
	axcolor = (20/255,175/255,245/255)	  # light blue
	hovercolor=(214/255,255/255,255/255)	# light grey
	lfp_all = lfp[id_all.reshape(-1,1) + np.arange(-timesteps,timesteps).reshape(1,-1)]
	ylims = np.mean(lfp_all) + 5*np.std(lfp_all)*np.array([-1,1])
	# ylims = np.max(np.abs(lfp_all-np.mean(lfp_all, axis=1, keepdims=True)))*0.75 * np.array([-1,1])

	# No need to pass oIn as parameter, but otherwise it needs to be defined after the object declaration
	def plot_ripples():
		for i,ax in enumerate(axes.flatten()):
			ax.cla()
			ax.set_yticklabels([])
			ax.set_xticklabels([])
			ax.set_xticks([])
			ax.set_yticks([])
			ax.set_facecolor('w')
		for i,ax in enumerate(axes.flatten()):
			if oIn.check_index(i):
				return
			disp_ind = i+oIn.index
			ini_window = np.maximum(id_all[disp_ind]-timesteps,0)
			end_window = np.minimum(id_all[disp_ind]+timesteps,len(lfp))
			lines = ax.plot(lfp[ini_window:end_window] - np.mean(lfp[ini_window:end_window]), 
				c=0.8*COLORS[from_swr_detector[disp_ind]], linewidth=0.5)
			ax.set_ylim(ylims)
			ax.set_xlim([-0.2*timesteps, 2*timesteps + timesteps*0.2])
			check_colors(oIn, i, ax)
	plot_ripples()

	# -- Buttons ----
	right_alignment = 0.945   # Right buttons position
	
	# Advance
	advance_ax = plt.axes([right_alignment, 0.60, 0.05, 0.05])	 # Advance button
	btn_advance = Button(advance_ax, f'Advance', color=axcolor, hovercolor=hovercolor)
	def advance(event):
		oIn.increase_index(events_in_screen)
		plot_ripples()
		fig.suptitle(f"Displaying ripples {oIn.index} to {oIn.n_og_len if (oIn.check_index(events_in_screen)) else oIn.index+events_in_screen } out of {oIn.n_og_len}")
		plt.draw()
		return 
	btn_advance.on_clicked(advance)

	# Regress
	regress_ax = plt.axes([right_alignment, 0.54, 0.05, 0.05])	# Go back button
	btn_regress = Button(regress_ax, f'Go back', color=axcolor, hovercolor=hovercolor)
	def regress(event):
		if (oIn.decrease_index(events_in_screen)):
			oIn.index=0
		plot_ripples()

		fig.suptitle(f"Displaying ripples {oIn.index} to {oIn.n_og_len if (oIn.check_index(events_in_screen)) else oIn.index+events_in_screen } out of {oIn.n_og_len}")
		plt.draw()
		return
	btn_regress.on_clicked(regress)

	# Discard all displayed events
	discard_ax = plt.axes([right_alignment, 0.48, 0.05, 0.05])
	btn_discard = Button(discard_ax, f'Discard all', color='#ff000032')
	def discard(event):
		oIn.set_keep_chunk(events_in_screen,False)
		for a,ax in enumerate(axes.flatten()):
			if oIn.check_index(a):
				break
			check_colors(oIn,a,ax)
		plt.draw()
		curated_intervals = oIn.intervals[oIn.keeps]
		format_predictions(os.path.join(saveas_folder, f'{file_name}_manual_inspection.txt'), curated_intervals, sf)
		return
	btn_discard.on_clicked(discard)
	
	# Keep all displayed events
	keep_ax = plt.axes([right_alignment, 0.42, 0.05, 0.05])
	btn_keep = Button(keep_ax, f'Keep all', color='#59ff3ab1')
	def keep(event):
		oIn.set_keep_chunk(events_in_screen,True)
		for a,ax in enumerate(axes.flatten()):
			if oIn.check_index(a):
				break
			check_colors(oIn,a,ax)
		plt.draw()
		curated_intervals = oIn.intervals[oIn.keeps]
		format_predictions(os.path.join(saveas_folder, f'{file_name}_manual_inspection.txt'), curated_intervals, sf)
		return
	btn_keep.on_clicked(keep)

	# Save events
	save_ax = plt.axes([right_alignment-0.0025, 0.30, 0.055, 0.08])
	btn_save = Button(save_ax, f'Save', color='#00c600b1')
	def save(event):
		curated_intervals = oIn.intervals[oIn.keeps]
		format_predictions(os.path.join(saveas_folder, f'{file_name}_manual_inspection.txt'), curated_intervals, sf)
		plt.close()
		# Plot and close
		id_win = np.arange(-win_size/2/1000*sf, win_size/2/1000*sf +1).astype(int).reshape(1,-1)
		events = lfp[id_win+id_all.reshape(-1,1)]
		b, a = scipy.signal.butter(2, 300, fs=sf, btype='lowpass')
		params['manual_inspection'] = True
		plot_curated_events(scipy.signal.filtfilt(b, a, events, axis=1), oIn.keeps, params, from_swr_detector=from_swr_detector)
		return
	btn_save.on_clicked(save)

	# Clicking
	def on_click(event):
		if event.button is MouseButton.LEFT:
			ax = event.inaxes
			if ax in axes:
				ax = event.inaxes
				# Indexes of the clicked subplot: (row,column)
				row_ind,col_ind=np.argwhere(axes==ax)[0]
				clicked_ind=(row_ind*int(events_in_screen/n_suby)+col_ind)

				if oIn.change_keep(clicked_ind):	# If out of bounds, close early, dont change color
					return
				check_colors(oIn,clicked_ind,ax)
				curated_intervals = oIn.intervals[oIn.keeps]
				format_predictions(os.path.join(saveas_folder, f'{file_name}_manual_inspection.txt'), curated_intervals, sf)
				plt.draw()
	plt.connect('button_press_event', on_click)
	plt.subplots_adjust(left=0.01,right=0.94,bottom=0.01,top=0.95,hspace=0.03,wspace=0.025)
	plt.show(block=True)

	# Return
	return np.array(oIn.keeps), from_swr_detector
