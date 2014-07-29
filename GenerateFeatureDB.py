from utilitiesSKM import *
from subprocess import call
import numpy as np

# Get all subject names.
subjectnames = check_output('ls -d */', shell=True)
subjectnames = subjectnames.rstrip().split('\n')

# Generate the empty feature DataFrame. 
### Put the column lables you'll need here!
columnlabels = ['species', 'seizure', 'early', 'electrodeNum', 'meanCorr', 'mean', 'varVar', 'meanVar']

### Edit this function to suit the features you want to extract from each clip.
def getFeatures(sample, dataPoint):
	adddata = [1 if 'Dog_' in sample else 0, temp[0], temp[1], len(temp[2])]
	corr = getCorr(temp)
	corr = corr.values[np.triu_indices_from(corr.values,1)]
	corr = corr[np.where(np.logical_not(np.isnan(corr)))] # Skips over any nan values.
	adddata.append(corr.mean())
	adddata.append(temp[2].mean().mean())
	adddata.append(temp[2].var().var())
	adddata.append(temp[2].var().mean())
	return adddata

# Dictionary for storing all processed clip information for dataframe creation.
traindata = {}
testdata = {}

# Gather the training data and convert to features.
for subject in subjectnames:
	print '\tProcessing ' + subject
	# Get the filenames to be trained for a given subject.
	filelist = getFileNames(subject, 'all')
	interfiles = filelist[0]
	ictfiles = filelist[1]
	testfiles = filelist[2]
	
	# For each interictal sample, extract the features.
	for sample in interfiles:
		temp = getDataPoint(sample)
		features = getFeatures(sample, temp)
		traindata[sample] = features

	# For each ictal sample, extract the features.
	for sample in ictfiles:
		temp = getDataPoint(sample)
		features = getFeatures(sample, temp)
		traindata[sample] = features

	# For each test sample, extract the features.
	for sample in testfiles:
		temp = getDataPoint(sample)
		features = getFeatures(sample, temp)
		testdata[sample] = features

# Use the extracted features to construct a pan-subject dataframe.
traindf = pd.DataFrame.from_dict(traindata, orient='index')
traindf.columns = columnlabels
traindf.to_csv('traindf.csv')

testdf = pd.DataFrame.from_dict(testdata, orient='index')
testdf.columns = columnlabels
testdf.to_csv('testdf.csv')
