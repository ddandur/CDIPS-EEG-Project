# To be run from the parent data directory, one level above the clip files.
# Fetches the data from the specified range of clips and returns as
# a normal list, where each element is a three member list containing
# a 0 or 1 (interictal, ictal), another 0 or 1 (1 if in first 15 seconds),
# then finally a pandas dataframe containing the data.

import pandas as pd
import scipy.io as io
from subprocess import check_output
from sys import exit
import matplotlib.pyplot as plt
import numpy as np

# Power Function: As a first pass, each power function calculates the power spectrum 
# for all electrodes, averages the spectrum across electrodes, 
# and finally bins the data into the selected frequency range.. This way the output 
# from each power function call is still a single real value. 
 
# The lowest sampling frequency is 400 Hz, which means the highest frequency 
# we should ask about should be 200 Hz. Thus lower_hz should be zero or above 
# and upper_hz should be 200 and below, with lower_hz < upperhz. 
# The input variables lower_hz and upper_hz should be integers.

# Calculate the real fast fourier transform on each electrode; 
# the way it's done here is to take modulus of each positive 
# complex frequency. The result pwr_array is still an array, a list with 
# num_el values where each element is the corresponding electrode's 
# values for the FFT squared, each value corresponding to power 
# at frequency 0 Hz, 1 Hz, etc up to the Nyquist frequency 

# Now average the power across all the electrodes; 
# ave_pwr is a single array of average power values at each frequency 
# among the electrodes 

# Now pick out the power values from the desired frequency range 
# and sum them. A nice feature for us is that the list index of the elements 
# in the ave_pwr list is also the frequency corresponding to that element in the list. 

def power(dataPoint):   
	num_el = len(dataPoint[2].T)
	freq_array = abs(np.fft.rfft(dataPoint[2]))
	pwr_array = freq_array**2
	sum_pwr = np.sum(pwr_array, axis = 1)
	ave_pwr = sum_pwr / float(num_el)
	return ave_pwr

# Specify a target subject (e.g. 'Dog_1') and, optionally, whether you
# want 'all' the data, 'train'ing data (ictal and interictal), or
# 'interictal', 'ictal', or 'test' alone.
# Returns a 3 member list. 
# [0] is a list of the interictal file names for the given subject.
# [1] is a list of the ictal file names for the given subject.
# [2] is a list of the test file names for the given subject.
# Assumes you are in the directory above the clip files!
def getFileNames(directory, typ = 'all'):

	call = ['','','']

	if (typ == 'all'):
		call[0] = check_output('find ' + directory + ' -name ' + '*_interictal*', shell=True)
		call[1] = check_output('find ' + directory + ' -name ' + '*_ictal*', shell=True)
		call[2] = check_output('find ' + directory + ' -name ' + '*_test*', shell=True)
		call[0] = call[0].split('\n')
		call[0].pop()
		call[0] = sorted(call[0])
		call[1] = call[1].split('\n')
		call[1].pop()
		call[1] = sorted(call[1])
		call[2] = call[2].split('\n')
		call[2].pop()
		call[2] = sorted(call[2])

	elif (typ == 'train'):
		call[0] = check_output('find ' + directory + ' -name ' + '*_interictal*', shell=True)
		call[1] = check_output('find ' + directory + ' -name ' + '*_ictal*', shell=True)
		call[0] = call[0].split('\n')
		call[0].pop()
		call[0] = sorted(call[0])
		call[1] = call[1].split('\n')
		call[1].pop()
		call[1] = sorted(call[1])

	elif (typ == 'interictal'):
		call[0] = check_output('find ' + directory + ' -name ' + '*_interictal*', shell=True)
		call[0] = call[0].split('\n')
		call[0].pop()
		call[0] = sorted(call[0])

	elif (typ == 'ictal'):
		call[1] = check_output('find ' + directory + ' -name ' + '*_ictal*', shell=True)
		call[1] = call[1].split('\n')
		call[1].pop()
		call[1] = sorted(call[1])

	elif (typ == 'test'):
		call[2] = check_output('find ' + directory + ' -name ' + '*_test*', shell=True)
		call[2] = call[2].split('\n')
		call[2].pop()
		call[2] = sorted(call[2])

	return call

# A simple function to retrieve info on a data point, given
# the path to get there from the current directory.
# Returns as a three member list.
# [0] 1 if ictal, 0 if not.
# [1] 1 if within the first 15 seconds of latency, 0 if not.
# [2] The dataframe containing the electrode data.
def getDataPoint(name):

	temp = io.loadmat(name)

	if '_ictal' in name:
		latency = temp['latency'][0]
		if (int(latency) < 15):
			temp = [1, 1, pd.DataFrame(temp['data']).T]
		else:
			temp = [1, 0, pd.DataFrame(temp['data']).T]
	# If not ictal, return the electrode dataframe.
	else:
		temp = [0, 0, pd.DataFrame(temp['data']).T]

	return temp

# Returns a specified portion of a certain type of data for a given subject.
# Specify the subject ('Dog_1'), type ('interictal'), the start (1 = 0001),
# and the end (178 = 0178), inclusive.
# Returns the data as a list of 3 member sub-lists. For the sub-lists:
# [0] 1 if ictal, 0 if not.
# [1] 1 if within the first 15 seconds of latency, 0 if not.
# [2] The dataframe containing the electrode data.
def getDataSlice(subject, typ, start, end):

	data = []

	# Input checking.
	if not (check_output('find . -name ' + subject, shell=True)):
		exit('The subject specified - ' + str(subject) + ' - was not found.')
	if not (typ == 'test' or typ == 'ictal' or typ == 'interictal'):
		exit('The type specified - ' + str(typ) + ' - was not \'ictal\', \'interictal\', or \'test\'.')
	startfile = './' + subject + '/' + subject + '_' + typ + '_segment_' + str(start).zfill(4) + '.mat'
	endfile = './' + subject + '/' + subject + '_' + typ + '_segment_' + str(end).zfill(4) + '.mat'
	if not (check_output('find . -wholename ' + startfile, shell=True)):
		exit('The starting clip value specified - ' + str(start) + ' - does not exist within the data.')
	if not (check_output('find . -wholename ' + endfile, shell=True)):
		exit('The starting clip value specified - ' + str(end) + ' - does not exist within the data.')
	if (int(start) > int(end)):
		exit('The range specified - ' + str(start) + ' to ' + str(end) + ' - is not valid.')
	
	# For the given range of file names.
	for x in range(int(start), int(end)+1):
		query = subject + '/' + subject + '_' + typ + '_segment_' + str(x).zfill(4) + '.mat'
		# Load the .mat file.
		temp = io.loadmat(query)
		# If ictal, set the latency value and construct the 3 member sub-list.
		if (typ == 'ictal'):
			latency = temp['latency'][0]
			if (int(latency) < 15):
				temp = [1, 1, pd.DataFrame(temp['data']).T]
			else:
				temp = [1, 0, pd.DataFrame(temp['data']).T]
		# If not ictal, return the electrode dataframe.
		else:
			temp = [0, 0, pd.DataFrame(temp['data']).T]
		
		data.append(temp)
	
	return data

# Accepts one of the typical training triplets and returns a 
# dataframe containing its correlations.
def getCorr(indata):
	
	corr = indata[2].corr()
	
	return corr

# Takes a correlation data frame and generates a heat map.
# If no output file is specified, will just display the plot. 
def imgCorr(indata, title, xaxis, yaxis, outfile=''):

	# Construct the plot.
	plt.imshow(indata, interpolation='nearest')
	plt.title(title)
	plt.xlabel(xaxis)
	plt.ylabel(yaxis)
	plt.colorbar()

	# Save if a file name is specified, otherwise just show.
	if (outfile):
		plt.savefig(title)
	else:
		plt.show()
	
	# Clean up.
	plt.clf()
	plt.cla()
