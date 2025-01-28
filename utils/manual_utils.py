# Imports
import numpy as np
# For interactive plots
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton

# Manual curation tool 
class saved_intervals:
	def __init__(self, original_intervals, from_swr):
		self.intervals = original_intervals
		self.n_og_len = len(original_intervals)
		self.index = 0
		self.keeps = self.n_og_len*[True]
		if type(from_swr) == np.ndarray:
			self.from_swr = from_swr
		else :
			self.from_swr = [False]*self.n_og_len
	# Check if a given number is out of bounds: True if out
	def check_index(self,n):
		# Positive or negative cases
		if (n>0 and self.index+n>=self.n_og_len) or (n<0 and self.index+n<0):
			return True
		return False
	# Increase the index the specified amount, return True if the desired amount is out of bound
	def increase_index(self,increment):
		if self.check_index(increment):
			return(True)
		self.index+=increment
	# Decrease the index the specified amount, return True if the desired amount is out of bound
	def decrease_index(self,decrement):
		if self.check_index(-decrement): # Calling check_index with a negative value manually
			return(True)
		self.index-=decrement
	def get_keep(self,n):
		return(self.keeps[self.index+n])
	def get_from_swr(self,n):
		return(self.from_swr[self.index+n])
	# Individual keep change, change keep to discard and viceversa for a single value
	def change_keep(self,ind):
		if (self.check_index(ind)): # True if out of bounds, returns True for excetion handling
			return True
		self.keeps[self.index+ind]= not (self.keeps[self.index+ind])
	# Multiple keep change, sets keep from index to index+number equal to value
	def set_keep_chunk(self,number,value):
		if self.check_index(number):
			self.keeps[self.index:self.n_og_len]=(self.n_og_len-self.index)*[value]
		else:
			self.keeps[self.index:self.index+number]=number*[value]

def check_colors(oIn,clicked_ind,ax):
	if not(oIn.get_keep(clicked_ind)):
		if oIn.get_from_swr(clicked_ind):
			ax.set_facecolor('#bdff007d')
		else:
			ax.set_facecolor('#ff000032')
	else:
		if oIn.get_from_swr(clicked_ind):
			ax.set_facecolor('#00ef0071')
		else:
			ax.set_facecolor('w')
	return
	
def get_click_th(event):
    if event.xdata<0:
        x=0
    elif event.xdata>1:
        x=1
    else:
        x=event.xdata
    return(x)

def format_predictions(path,preds,d_sf):
    ''' 
    format_predictions(path,preds,d_sf) 

    Writes a .txt with the initial and end times of events in seconds 

    Inputs:
    -------
        path       str, absolute path of the file that will be created
        preds      (1,n_events) np.array with the initial and end timestamps of the events
        d_sf        int, sampling frequency of the data

    A Rubio, LCN 2023
    '''
    f=open(path,'w')

    preds=preds/d_sf
    for pred in preds:
        f.write(str(pred))
        f.write('\n')
    f.close()
    return  