# Imports
import os, sys, scipy
import numpy as np
# Directory of project
path_project = os.getcwd()
path_data = os.path.join(path_project, 'data')
# Our code
sys.path.insert(1, path_project)
sys.path.insert(1, os.path.join(path_project, 'utils'))
import load_utils, ripmap

# ==================================================================
# 			INITIALIZE
# ==================================================================

# Analyse one mat file
file_name = 'ER20_micro1_epoch1.mat'
# Load LFP, channel detection and srate
session_dict = load_utils.read_matlab_v73_files(os.path.join(path_data,file_name))
sf = float(session_dict['srate'][0,0])
channel = int(session_dict['detectionChannel'][0,0]) -1 # -1 if saved from matlab
lfp = session_dict['lfp'][channel,:].flatten()
# Load detections
detections_dict = scipy.io.loadmat(os.path.join(path_data,file_name[:-4]+'_automatic_detections.mat'))
times = { 'swrs': detections_dict['HFOs'].flatten(), 
		  'ieds': detections_dict['IEDs'].flatten(),
		  'id_fps': (detections_dict['flagged'].flatten()-1).astype(int) if 'flagged' in detections_dict.keys() else []
		 }

# Parameters
win_size_show = 0.100
win_size_umap = 0.010
do_detrend = True
do_zscore = False
list_n_neighbors = [10, 50, 100]
list_min_dists = [0.0, 0.1, 0.2]
intrinsic_dimension = 4
saveas_folder = os.path.join(path_project, 'figures')
n_elements = 30
n_axis_bins = 9
axis_method = 'centroids'
do_axis_grid = False
fp_separately = False
selected_hfos, selected_ieds, events, params = ripmap.event_curation(lfp, sf, times, 
								win_size_show=win_size_show, win_size_umap=win_size_umap, 
								do_detrend=do_detrend, do_zscore=do_zscore, 
								list_n_neighbors=list_n_neighbors, list_min_dists=list_min_dists, 
								n_elements=n_elements, intrinsic_dimension=intrinsic_dimension,
								n_axis_bins=n_axis_bins, do_axis_grid=do_axis_grid, 
								axis_method=axis_method, plot_fp_separately=fp_separately,
								file_name=file_name, saveas_folder=saveas_folder, save_format='png')

# Final manual inspection - optional
ids_keep = ripmap.manual_inspection(lfp, sf, times['swrs'][selected_hfos], times['ieds'][selected_ieds], 
								params, events_in_screen=50, win_size=200,
								file_name=file_name, saveas_folder=saveas_folder)

# Save
mdict = {'selected_hfos':selected_hfos, 
		 'selected_ieds':selected_ieds, 
		 'curated_manually':ids_keep,
		 'curates_times': np.append(times['swrs'][selected_hfos], times['ieds'][selected_ieds])[ids_keep],
		 'events':events, 
		 'params':params}
if not os.path.exists(os.path.join(path_project, 'results')): os.mkdir(os.path.join(path_project, 'results'))
scipy.io.savemat(os.path.join(path_project, 'results', f'{file_name}_curated_labels.mat'), mdict)

