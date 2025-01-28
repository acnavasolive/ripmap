# Imports
import os, sys, scipy
import numpy as np
# Directory of project
path_project = os.getcwd()
path_data = os.path.join(path_project, 'data')
# Our code
sys.path.insert(1, path_project)
sys.path.insert(1, os.path.join(path_project, 'utils'))
import load_utils, toolbox

# ==================================================================
# 			INITIALIZE
# ==================================================================

# Analyse one mat file
file_name = 'ER20_micro1.mat'
# Load variables
file_dict = load_utils.read_matlab_v73_files(os.path.join(path_data,file_name))
sf = file_dict['srate'][0,0]
lfp = file_dict['lfp'].flatten()
times = { 'swrs': file_dict['HFOs']['peaks'].flatten(), 
		  'ieds': file_dict['IEDs'].flatten(),
		  'id_fps': (file_dict['HFOs']['flagged'].flatten()-1).astype(int) if 'flagged' in file_dict['HFOs'].keys() else []
		 }

# Parameters
win_size_show = 0.100
win_size_umap = 0.020
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
selected_hfos, selected_ieds, events, params = toolbox.event_curation(lfp, sf, times, 
								win_size_show=win_size_show, win_size_umap=win_size_umap, 
								do_detrend=do_detrend, do_zscore=do_zscore, 
								list_n_neighbors=list_n_neighbors, list_min_dists=list_min_dists, 
								n_elements=n_elements, intrinsic_dimension=intrinsic_dimension,
								n_axis_bins=n_axis_bins, do_axis_grid=do_axis_grid, 
								axis_method=axis_method, plot_fp_separately=fp_separately,
								file_name=file_name, saveas_folder=saveas_folder, save_format='png')

# Final manual inspection - optional
ids_keep = toolbox.manual_inspection(lfp, sf, times['swrs'][selected_hfos], times['ieds'][selected_ieds], 
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

