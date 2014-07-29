# Module import.
import pandas as pd
from utilitiesSKM import *
from datetime import datetime
import numpy as np

print str(datetime.now())

# Import the traindf and testdf databases.
traindf = pd.read_csv('traindf.csv', index_col=0)
testdf = pd.read_csv('testdf.csv', index_col=0)

### Name your feature(s).
featureName = 'dummyFeature'

# Accumulate the feature for the training data,
# then add to the traindf dataframe and save.
featureVals = np.array([])

for sample in traindf.index:
	temp = getDataPoint(sample)
	### Extract your feature(s) here.
	feature = temp[2][0][0] # likely some operation on the data in temp[2]
	featureVals = np.append(featureVals, feature)

traindf[featureName] = pd.Series(featureVals, index=traindf.index)
traindf.to_csv('traindf.csv')

print str(datetime.now())

featureVals = np.array([])

for sample in testdf.index:
	temp = getDataPoint(sample)
	### Extract your feature(s) here.
	feature = temp[2][0][0] # likely some operation on the data in temp[2]
	featureVals = np.append(featureVals, feature)

testdf[featureName] = pd.Series(featureVals, index=testdf.index)
testdf.to_csv('testdf.csv')

print str(datetime.now())
	
