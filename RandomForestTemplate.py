# Haven't re-factored yet to make as easy to use as possible.
# However, I have gone through and highlighted areas of interest
# for modifiation and tweaking with comments having '###'.

# Import requisite modules.
from utilitiesSKM import *
from datetime import datetime
from random import sample
from subprocess import call
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# For logging info about run, make subdirectory and
# start a log file.
startrun = datetime.now()
date = str(startrun)[0:10]

# Set the directory where you desire to have the output
# and log files stored.
### Set the output directory.
OUTPUTdir = '../' + date + '_RFattempt1/'
call('mkdir ' + OUTPUTdir, shell=True)
statuslog = open(OUTPUTdir + 'RFLog.txt', 'a')
statuslog.write(date + '\nFirst attempt at random forest classification.\n\n')

# Get all subject names.
subjectnames = check_output('ls -d */', shell=True)
subjectnames = subjectnames.rstrip().split('\n')

### This section determines the training and validation set.
### Modify as per your liking.
# Remove a random set of 3 subjects to validate later.
# exclude = list(reversed(sorted(sample(range(12), 3))))
exclude = [11, 5, 0]
excluded = []
for x in exclude:
	excluded.append(subjectnames.pop(x))
str(sorted(excluded))
statuslog.write('Subjects excluded:\n')
for x in excluded:
	statuslog.write('\t' + x + '\n')
statuslog.write('\n\nSubjects included:\n')
for x in subjectnames:
	statuslog.write('\t' + x + '\n')

statuslog.write('\nStarting analysis...\n')

# Generate the empty feature DataFrame. 
### Put the column lables you'll need here!
columnlabels = ['species', 'seizure', 'early', 'electrodeNum', 'meanCorr', 'mean', 'varVar', 'meanVar']

### Put the names of the features you'll be using to train here!
featurelabels = np.array(['species', 'meanCorr', 'mean', 'varVar', 'meanVar'])

dictdata = {}

# Gather the training data and convert to features.
for subject in subjectnames:
	statuslog.write('\tProcessing ' + subject + ' at ' + str(datetime.now()) + '\n')
	print('\tProcessing ' + subject + ' at ' + str(datetime.now()) + '\n')
	# Get the filenames to be trained for a given subject.
	filelist = getFileNames(subject, 'train')
	interfiles = filelist[0]
	ictfiles = filelist[1]
	
	### These two for loops should be modified to get the
	### features desired.
	# For each interictal sample, extract the features.
	for sample in interfiles:
		temp = getDataPoint(sample)
		adddata = [1 if 'Dog_' in sample else 0, temp[0], temp[1], len(temp[2])]
		corr = getCorr(temp)
		adddata.append(corr.mean().mean())
		adddata.append(temp[2].mean().mean())
		adddata.append(temp[2].var().var())
		adddata.append(temp[2].var().mean())
		dictdata[sample] = adddata
	# For each ictal sample, extract the features.
	for sample in ictfiles:
		temp = getDataPoint(sample)
		adddata = [1 if 'Dog_' in sample else 0, temp[0], temp[1], len(temp[2])]
		corr = getCorr(temp)
		adddata.append(corr.mean().mean())
		adddata.append(temp[2].mean().mean())
		adddata.append(temp[2].var().var())
		adddata.append(temp[2].var().mean())
		dictdata[sample] = adddata

# Use the extracted features to construct a pan-subject dataframe.
traindf = pd.DataFrame.from_dict(dictdata, orient='index')
traindf.columns = columnlabels
traindf.to_csv(OUTPUTDIR + 'IncludedDF.csv')
del dictdata

statuslog.write('Feature generation finished in ' + str(datetime.now()-startrun) + '.\nProceeding to training...\n\n')
trainingstart = datetime.now()

### If you want to try a different ML algorithm, change this part!
# Train random forest classifier for predicting if seizure or not.
clf1 = RandomForestClassifier()
clf1.fit(traindf[featurelabels], traindf['seizure'])

# Save the classifier. Use joblib.load() when desired to retrieve.
saveclf1 = OUTPUTdir + 'RFClassifierseizure.pkl'
joblib.dump(clf1, saveclf1)

statuslog.write('Generated RFseizure:\n' + str(clf1) + '\n\n')

### If you want to try a different ML algorithm, change this part!
# Train random forest classifier for predicting if early or not.
clf2 = RandomForestClassifier()
clf2.fit(traindf[featurelabels], traindf['early'])

# Save the classifier. Use joblib.load() when desired to retrieve.
saveclf2 = OUTPUTdir + 'RFClassifierEarly.pkl'
joblib.dump(clf2, saveclf2)

statuslog.write('Generated RFearly:\n' + str(clf2) + '\n\n')
statuslog.write('Analysis and training completed in ' + str(datetime.now()-startrun) + '.\n')
statuslog.write('The classifiers have been exported to ' + saveclf1 + ' and ' + saveclf2 + ' via joblib. Use joblib.load() targeted at the parent .pkl file to recall it!\n\n')

# Plot and save the importances of the features as calculated by the RFC
# for classifying seizure clips.
# This part is optional but informative.
importances = clf1.feature_importances_
sorted_idx = np.argsort(importances)
padding = np.arange(len(featurelabels)) + 0.5
plt.barh(padding, importances[sorted_idx], align='center')
plt.yticks(padding, featurelabels[sorted_idx])
plt.xlabel("Relative Importance")
plt.title("Variable Importance for Seizure Prediction")
plt.savefig(OUTPUTdir + 'FeatureImportanceSeizure.png')
plt.clf()
plt.cla()

# Plot and save the importances of the features as calculated by the RFC
# for classifying early onset.
# This part is optional but informative.
importances = clf2.feature_importances_
sorted_idx = np.argsort(importances)
padding = np.arange(len(featurelabels)) + 0.5
plt.barh(padding, importances[sorted_idx], align='center')
plt.yticks(padding, featurelabels[sorted_idx])
plt.xlabel("Relative Importance")
plt.title("Variable Importance for Early Prediction")
plt.savefig(OUTPUTdir + 'FeatureImportanceEarly.png')
plt.clf()
plt.cla()

print 'Testing excluded subjects.'
statuslog.write('Now moving on to test against excluded subjects...\n\n')

validfeaturestart = datetime.now()

dictdata = {}

# This part grabs the features, as above, for the excluded data.
# This allows for subsequent validation.
for subject in excluded:
	statuslog.write('\tProcessing ' + subject + ' at ' + str(datetime.now()) + '\n')
	print('\tProcessing ' + subject + ' at ' + str(datetime.now()) + '\n')
	# Get the filenames to be trained for a given subject.
	filelist = getFileNames(subject, 'train')
	interfiles = filelist[0]
	ictfiles = filelist[1]
	
	### Make sure you extract the same features as for the training!
	# For each interictal sample, extract the features.
	for sample in interfiles:
		temp = getDataPoint(sample)
		adddata = [1 if 'Dog_' in sample else 0, temp[0], temp[1], len(temp[2])]
		corr = getCorr(temp)
		adddata.append(corr.mean().mean())
		adddata.append(temp[2].mean().mean())
		adddata.append(temp[2].var().var())
		adddata.append(temp[2].var().mean())
		dictdata[sample] = adddata
	# For each ictal sample, extract the features.
	for sample in ictfiles:
		temp = getDataPoint(sample)
		adddata = [1 if 'Dog_' in sample else 0, temp[0], temp[1], len(temp[2])]
		corr = getCorr(temp)
		adddata.append(corr.mean().mean())
		adddata.append(temp[2].mean().mean())
		adddata.append(temp[2].var().var())
		adddata.append(temp[2].var().mean())
		dictdata[sample] = adddata

# Use the extracted features to construct a pan-subject dataframe.
valdf = pd.DataFrame.from_dict(dictdata, orient='index')
valdf.columns = columnlabels
valdf.to_csv(OUTPUTDIR + '_RFattempt/ExcludedDF.csv')
del dictdata

# Log the progress.
statuslog.write('Feature generation for validation set finished in ' + str(datetime.now()-validfeaturestart) + '.\nProceeding to validation...\n\n')

# Calculate the accuracies by testing against the data that was reserved / excluded.
acc1 = clf1.score(valdf[featurelabels], valdf['seizure'])
acc2 = clf2.score(valdf[featurelabels], valdf['early'])

# Log the accuracies.
statuslog.write('Accuracy for identification of seizure was ' + str(acc1)[0:6] + ' and accuracy for identification of early onset was ' + str(acc2)[0:6] + '.\n\n')

print 'Generating features for test set...'

testfeaturestart = datetime.now()

# Now we need to get all the test data and extract it's features.
subjectnames.extend(excluded)
dictdata = {}
teststart = datetime.now()
for subject in subjectnames:
	statuslog.write('\tProcessing ' + subject + ' at ' + str(datetime.now()) + '\n')
	print('\tProcessing ' + subject + ' at ' + str(datetime.now()) + '\n')
	# Get the filenames to be trained for a given subject.
	filelist = getFileNames(subject, 'test')
	testfiles = filelist[2]
	
	### Extract test set features. Make sure these match-up
	### with the training and validation features!
	# For each interictal sample, extract the features.
	for sample in testfiles:
		temp = getDataPoint(sample)
		adddata = [1 if 'Dog_' in sample else 0, '', '', len(temp[2])]
		corr = getCorr(temp)
		adddata.append(corr.mean().mean())
		adddata.append(temp[2].mean().mean())
		adddata.append(temp[2].var().var())
		adddata.append(temp[2].var().mean())
		dictdata[sample] = adddata

# Use the extracted features to construct a pan-subject dataframe.
testdf = pd.DataFrame.from_dict(dictdata, orient='index')
testdf.columns = columnlabels
testdf.to_csv(OUTPUTdir + 'TestDF.csv')
del dictdata

# Log progress.
statuslog.write('Feature generation for test set finished in ' + str(datetime.now()-testfeaturestart) + '.\nProceeding to generate predictions...\n\n')

# Generate the prediction data.
predictions = []
testseizureprob = []
testearlyprob = []
testseizureprob = clf1.predict_proba(testdf[featurelabels])
testearlyprob = clf2.predict_proba(testdf[featurelabels])

# Merge the predictions together into a single list.
predictions = zip(testdf.index, testseizureprob, testearlyprob)
for x in range(len(predictions)):
    predictions[x] = [predictions[x][0], predictions[x][1][1], predictions[x][2][1]]

# Output the merged predictions into a csv for submission.
fileout = open(OUTPUTdir + 'Predictions.csv', 'a')
fileout.write('clip,seizure,early\n')
for x in predictions:
	fileout.write(x[0] + ',' + str(x[1]) + ',' + str(x[2]) + '\n')

# Final log.
statuslog.write('Total run time finished in ' + str(datetime.now()-startrun) + '.\n')

# Ready to submit!

