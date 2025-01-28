import numpy as np

# Imports for ABIDs (intrinsic dimension)
from scipy.spatial import cKDTree
from tqdm import tqdm

# Persistent homology
from ripser import ripser
from persim import plot_diagrams
from sklearn.metrics import pairwise_distances

# ==== UTILS TOPOLOGY =======================================================

def compute_abids(base_signal=None, n_neigh=50, verbose=True):
	'''Compute dimensionality of data array according to (arXiv:2006.12880)
	
	Parameters:
	-----------
	    base_signal (numpy array): 
	        array containing the signal one wants to estimate dimensionality.
	    n_neigh (int): 
	        number of neighbours used to compute angle.                     
	Returns:
	--------
	    (array): 
	        array containing the estimated dimensionality for each point in the
	        cloud.
	    
	'''

	def abid(X,k,x,search_struct,offset=1):
		neighbor_norms, neighbors = search_struct.query(x,k+offset)
		neighbors = X[neighbors[offset:]] - x
		normed_neighbors = neighbors / neighbor_norms[offset:,None]
		# Original publication version that computes all cosines
		# coss = normed_neighbors.dot(normed_neighbors.T)
		# return np.mean(np.square(coss))**-1
		# Using another product to get the same values with less effort
		para_coss = normed_neighbors.T.dot(normed_neighbors)
		return k**2 / np.sum(np.square(para_coss))

	search_struct = cKDTree(base_signal)
	if verbose:
		return np.array([
		    abid(base_signal,n_neigh,x,search_struct)
		    for x in tqdm(base_signal, desc="abids", leave=False)
		])
	else:
		return np.array([abid(base_signal,n_neigh,x,search_struct) for x in base_signal])


def compute_diagrams(events, max_B):
	# Get maximum radius
	print('Computing pairwise distances...\t\t', end='\r')
	ds = pairwise_distances(events)
	thresh = int(np.ceil(np.nanmax(ds)))
	# Compute diagrams
	print('Computing diagrams...\t\t\t\t', end='\r')
	diagrams = ripser(events, maxdim=max_B, thresh=thresh)['dgms']
	# Change infinity to thresh
	diagrams[0][-1,1] = np.nanmax(ds)
	return diagrams, ds, thresh


def compute_dense_diagrams(diagrams, ds, n_elements=None):
	# - All distances (without repetitions)
	ds_tril = ds[np.tril_indices_from(ds, k=-1)] # lower-triangular indexes
	max_d = np.max(ds_tril)
	# - Initialize
	dense_diagrams = np.copy(diagrams)
	# - Go through all betti numbers and bars
	n_hist = 1000
	for Hi in range(len(diagrams)):
		xhist = np.linspace(0,max_d,n_hist+1)
		yhist = np.histogram(ds[np.tril_indices_from(ds, k=-1)], xhist)[0]
		xhist = xhist[1:] - (xhist[1]-xhist[0])/2.
		cumsum_distances = np.cumsum(yhist)
		cumsum_distances = cumsum_distances / np.max(cumsum_distances)

		# - Which bars to compute it from
		n_bars = diagrams[Hi].shape[0]
		bars = np.arange(diagrams[Hi].shape[0]) if n_elements == None else np.arange(n_bars-n_elements,n_bars)
		for bar in bars:
			# Start and end radius of bars
			st = diagrams[Hi][bar][0]
			en = diagrams[Hi][bar][1]
			# Take value from histogram
			dense_diagrams[Hi][bar][0] = cumsum_distances[int(st/max_d*(n_hist-1))]
			dense_diagrams[Hi][bar][1] = cumsum_distances[int(en/max_d*(n_hist-1))]

	return dense_diagrams, cumsum_distances


# ==== UTILS VISUALIZATION =======================================================

def centroid_curve(xs1, ys1, xs2, ys2):

	# Centroid 1
	x1, y1 = np.mean(xs1), np.mean(ys1)
	# Centroid 2
	x2, y2 = np.mean(xs2), np.mean(ys2)
	# Make 1 be the one with the lowest x
	xs = np.array([x1, x2])
	ys = np.array([y1, y2])
	i1, i2 = np.argsort(xs)
	x1 = xs[i1]
	y1 = ys[i1]
	x2 = xs[i2]
	y2 = ys[i2]
	# Cuadratic term
	a2 = 0.
	# Slope
	a1 = (y2-y1) / (x2-x1)
	# Intercept
	a0 = y1 - a1*x1
	return a2, a1, a0

def fit_axis(x, a2, a1, a0):
	return a0 + a1*x + a2*np.power(x,2)

def axis_tangent(x0, r, a2, a1, a0):
	# Slope
	m = a1 + 2*a2*x0 
	# x projection
	x1 = x0 + r*np.cos(np.arctan(m))
	# y projection
	y0 = fit_axis(x0, a2, a1, a0)
	y1 = y0 + r*np.sin(np.arctan(m))
	#print(f'r = {np.sqrt((x0-x1)**2 + (y0-y1)**2)}')
	return np.array([x1, y1])

def divide_axis(rini, rend, n_bins, a2, a1, a0):
	# Total length of axis
	L = 0
	r = 0.01
	x0, y0 = rini
	while x0 < rend[0]:
		x0, y0 = axis_tangent(x0, r, a2, a1, a0)
		L += r

	# Distance between points
	r = L/n_bins
	# Divide axis in n_bins points
	xdivs = [rini[0]]
	for i in range(n_bins):
		xnext, ynext = axis_tangent(xdivs[-1], r, a2, a1, a0)
		xdivs.append(xnext)
		#print( np.sqrt((xnext-xdivs[-1])**2 + (ynext-fit_axis(xdivs[-1],*popt))**2) )
	xdivs = np.array(xdivs)

	return xdivs, fit_axis(xdivs, a2, a1, a0)

def project_to_curve(x, y, popt):
	# Initialize
	xproj, yproj = np.nan*np.ones_like(x), np.nan*np.ones_like(y)
	all_xs = np.linspace(np.min(x)-2, np.max(x)+2, 2000)
	# Go through all points in UMAP
	for i, (xi, yi) in enumerate((zip(x, y))):
		# Create a specific function that is the distance 
		# between the curve fit_axis and the point (xi,yi)
		fun = lambda x: np.sqrt( (xi - x)**2 + (yi - fit_axis(x, *popt))**2 )
		# Minimize the function
		# res = scipy.optimize.minimize(fun, np.array(xi,yi))
		# xproj[i] = res.x
		# yproj[i] = fit_axis(xproj, *popt)[0]
		xproj[i] = all_xs[np.argmin(fun(all_xs))]
		yproj[i] = fit_axis(xproj[i], *popt)
	return xproj, yproj

def bin_events_in_axis(xproj, xdivs):

	event_bins = np.zeros_like(xproj).astype(int)
	# Include each xproj into its xhist interval
	for i in range(len(xdivs)-1):
		event_bins[(xproj>=xdivs[i]) & (xproj<xdivs[i+1])] = i
	return event_bins

def make_projected_histogram(xproj, xdivs, n_bins, putative_type, norm_hist=True):
	
	types = np.unique(putative_type)
	n_type = len(np.unique(putative_type))

	# Define xhist axis
	xhist = np.linspace(np.min(xproj), np.max(xproj), n_bins)
	dxhist = xhist[1]-xhist[0]
	# Initialize
	yhists = np.zeros((n_type, len(xhist))) 
	# Include each xproj into its xhist interval
	for i in range(len(xhist)):
		for itype in range(n_type):
			yhists[itype][i] = np.sum( (xproj[putative_type==types[itype]]>=xdivs[i]) & (xproj[putative_type==types[itype]]<xdivs[i+1]) )
	# Normalize
	if norm_hist:
		for itype in range(n_type):
			yhists[itype] = yhists[itype]/np.sum(yhists[itype])
	return yhists