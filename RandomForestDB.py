# Haven't re-factored yet to make as easy to use as possible.
# However, I have gone through and highlighted areas of interest
# for modifiation and tweaking with comments having '###'.

# Import requisite modules.
from utilitiesSKM import *
from datetime import datetime
from random import random
from subprocess import call
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import numpy as np
import re

# For logging info about run, make subdirectory and
# start a log file.
startrun = datetime.now()
date = str(startrun)[0:10]

# Set the directory where you desire to have the output
# and log files stored.
### Set the output directory.
OUTPUTdir = '../' + date + 'DB1AllTrain/'
call('mkdir ' + OUTPUTdir, shell=True)
statuslog = open(OUTPUTdir + 'Log.txt', 'a')
statuslog.write(date + '\nUse DB for faster pipeline, use all data to train.\n\n')

# Load traindf.
traindf = pd.read_csv('traindf.csv', index_col = 0)

### This section determines the training and validation set.
### Modify as per your liking.
cutoff = 0.25
valdf = ''
# Transpose traindf for easier processing.
traindf = traindf.T
count = 0
for x in traindf.columns:
	chance = random()
	if (chance <= cutoff):
		if (type(valdf) != pd.core.series.Series and type(valdf) != pd.core.frame.DataFrame):
			valdf = traindf.pop(x)
		else:
			valdf = pd.concat([valdf, traindf.pop(x)], axis=1)
	# For the sake of keeping track of progress.
	count += 1
	if (count % 500 == 0):
		print count

# Convert both back to normal orientation.
traindf = traindf.T
valdf = valdf.T

statuslog.write('\nStarting analysis...\n')

### Put the names of the features you'll be using to train here!
featurelabels = np.array(['meanCorr', 'mean', 'varVar', 'meanVar', 'delta', 'theta', 'alpha', 'beta', 'lowGamma', 'highGamma'])

# Save the employed training and validation sets.
traindf.to_csv(OUTPUTdir + 'TrainingSet.csv')
valdf.to_csv(OUTPUTdir + 'ValidationSet.csv')

statuslog.write('Feature generation finished in ' + str(datetime.now()-startrun) + '.\nProceeding to training...\n\n')
trainingstart = datetime.now()

### If you want to try a different ML algorithm, change this part!
# Train random forest classifier for predicting if seizure or not.
clf1 = RandomForestClassifier(n_estimators=100)
clf1.fit(traindf[featurelabels], traindf['seizure'])

# Save the classifier. Use joblib.load() when desired to retrieve.
saveclf1 = OUTPUTdir + 'RFClassifierseizure.pkl'
joblib.dump(clf1, saveclf1)

statuslog.write('Generated RFseizure:\n' + str(clf1) + '\n\n')

### If you want to try a different ML algorithm, change this part!
# Train random forest classifier for predicting if early or not.
clf2 = RandomForestClassifier(n_estimators=100)
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

# If a validation set was generated:
if (cutoff > 0):
	print 'Validating on excluded subjects.'
	statuslog.write('Now moving on to test against excluded subjects...\n\n')
	validfeaturestart = datetime.now()

	# Log the progress.
	statuslog.write('Feature generation for validation set finished in ' + str(datetime.now()-validfeaturestart) + '.\nProceeding to validation...\n\n')

	# Calculate the accuracies by testing against the data that was reserved / excluded.
	acc1 = clf1.score(valdf[featurelabels], valdf['seizure'])
	acc2 = clf2.score(valdf[featurelabels], valdf['early'])

	# Log the accuracies.
	print 'Accuracy for identification of seizure was ' + str(acc1)[0:6] + '\nAccuracy for identification of early onset was ' + str(acc2)[0:6]
	statuslog.write('Accuracy for identification of seizure was ' + str(acc1)[0:6] + ' and accuracy for identification of early onset was ' + str(acc2)[0:6] + '.\n\n')

	# Next calculate the ROC values. Log those values, and save the plots.
	# First, we need to make predictions over the validation set with the classifiers.
	valseizureprob = clf1.predict_proba(valdf[featurelabels])
	valearlyprob = clf2.predict_proba(valdf[featurelabels])
	# Then, we need to plug them into the ROC curve calculation.
	valsfpr, valstpr, _ = roc_curve(valdf['seizure'], valseizureprob[:,1])
	valefpr, valetpr, _ = roc_curve(valdf['early'], valearlyprob[:,1])
	# Plot and save.
	plt.plot(valsfpr, valstpr)
	plt.savefig(OUTPUTdir + 'seizureROC.png')
	plt.clf()
	plt.cla()
	plt.plot(valefpr, valetpr)
	plt.savefig(OUTPUTdir + 'earlyROC.png')
	plt.clf()
	plt.cla()
	# Get areas under the curve.
	seizureAUC = auc(valsfpr, valstpr)
	earlyAUC = auc(valefpr, valetpr)
	# Report / log.
	print 'Seizure AUC = ' + str(seizureAUC) + '\nEarly AUC = ' + str(earlyAUC)
	statuslog.write('Seizure AUC = ' + str(seizureAUC) + '\nEarly AUC = ' + str(earlyAUC) + '\n\n')

statuslog.write('Proceeding to generate predictions...\n\n')

# Load test database features.
testdf = pd.read_csv('testdf.csv', index_col = 0)

# Generate the prediction data.
predictions = []
testseizureprob = []
testearlyprob = []
testseizureprob = clf1.predict_proba(testdf[featurelabels].values)
testearlyprob = clf2.predict_proba(testdf[featurelabels].values)

# Merge the predictions together into a single list.
predictions = zip(testdf.index, testseizureprob, testearlyprob)
for x in range(len(predictions)):
    predictions[x] = [predictions[x][0], predictions[x][1][1], predictions[x][2][1]]

# Get rid of the directory and convert the numeric identifier back
# from the 4 digit format we originally converted it to.
findnameRE = re.compile('.+/(.+\.mat)')
findnumRE = re.compile('segment_(\d{4})')
subnumRE = re.compile('segment_\d{4}')
for x in range(len(predictions)):
	tempname = re.search(findnameRE, predictions[x][0]).group(1)
	oldnum = re.search(findnumRE, predictions[x][0]).group(1)
	newnum = 'segment_' + str(int(oldnum))
	newname = re.sub(subnumRE, newnum, tempname)
	predictions[x][0] = newname

# Output the merged predictions into a csv for submission.
fileout = open(OUTPUTdir + 'Predictions.csv', 'a')
fileout.write('clip,seizure,early\n')
for x in predictions:
	fileout.write(x[0] + ',' + str(x[1]) + ',' + str(x[2]) + '\n')

# Final log.
statuslog.write('Total run time finished in ' + str(datetime.now()-startrun) + '.\n')

# Ready to submit!

