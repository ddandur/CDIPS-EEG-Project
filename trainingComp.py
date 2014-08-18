from ecog_tools import *
import pandas as pd
import numpy as np
import sklearn
import sklearn.ensemble
import scipy.stats
from random import shuffle
import re

numRE = re.compile('_(\d{4})')

total_test_df = pd.DataFrame()

subjects = []
for i in range(1,9):
    subject = 'Patient_%i'%(i)
    subjects.append(subject)
for i in range(1,5):
    subject = 'Dog_%i'%(i)
    subjects.append(subject)

########

predicted_aucRF = 0
predicted_aucAB = 0
predicted_aucGB = 0

#subjects = ['Patient_1']

for subject in subjects:
    

    # Create a list of all .mat file names.
    ##### This is where you decide which features to train on!
    #    training_functions  = [early_ictal,ictal,hgammapower]

    #### using this training_functions will run with every single feature in the featuredb csv files!
    #    uncomment the previous training_functions with the specific functions you want to train on if you don't want to
    #    include every single function (remember to include ictal and early_ictal) and remove this training_functions

    training_functions = [eval(i) for i in pd.DataFrame.from_csv('%s_functions'%(subject)).index]

    #print training_functions

    feature_dataframe = extract_features(training_functions,'%s_features'%(subject),'%s_functions'%(subject))
    function_dataframe = pd.DataFrame.from_csv('%s_functions'%(subject))

    eeg_mat_fnames = list(feature_dataframe.index)

    # Seperate files by type.
    test_files = [file for file in eeg_mat_fnames if is_test(file)]

    # Shuffle files to seperate for training and validation.
    ictal_files = [file for file in eeg_mat_fnames if is_ictal(file)]
    early_ictal_files = [file for file in ictal_files if is_early_ictal(file)]
    ictal_files = [file for file in ictal_files if not is_early_ictal(file)]
    shuffle(ictal_files)
    shuffle(early_ictal_files)
    inter_ictal_files =[file for file in eeg_mat_fnames if is_inter_ictal(file)]
    shuffle(inter_ictal_files)

    # Break up into validation and training set.
    # v_percent of the data is withheld for validation.
    v_percent = 0.2

    train_set = early_ictal_files[int(v_percent*len(early_ictal_files)):]
    train_set += ictal_files[int(v_percent*len(ictal_files)):]
    train_set += inter_ictal_files[int(v_percent*len(inter_ictal_files)):]

    # Set a validation set even if you're training with all the data ... just because.
    if (v_percent == 0.0):
        v_percent = 0.2

    val_set = early_ictal_files[:int(v_percent*len(early_ictal_files))]
    val_set += ictal_files[:int(v_percent*len(ictal_files))]
    val_set += inter_ictal_files[:int(v_percent*len(inter_ictal_files))]
    
    shuffle(val_set)
    shuffle(train_set)

    training_df = feature_dataframe.loc[train_set]
    val_df = feature_dataframe.loc[val_set]
    test_df = feature_dataframe.loc[test_files]

    classifier_labels = ['ictal','early_ictal']
    feature_labels = list(training_df.drop(classifier_labels, axis=1).columns)
    column_labels = list(training_df.columns)

    training_df = training_df.dropna()
    y_test = []
    y_score = []
    fpr = [0,0]
    tpr = [0,0]
    # calculate ictal and early ictal ROC curves i=0 is ictal, i=1 early ictal
    classifierictal = sklearn.ensemble.RandomForestClassifier(n_estimators=100, n_jobs=-1)
    classifierictal.fit(training_df[feature_labels],training_df['ictal'])
    y_test.append(np.array(val_df['ictal']))
    y_score.append(classifierictal.predict_proba(val_df[feature_labels]).T[1])
    classifierearly = sklearn.ensemble.RandomForestClassifier(n_estimators=100, n_jobs=-1)
    classifierearly.fit(training_df[feature_labels],training_df['early_ictal'])
    y_test.append(np.array(val_df['early_ictal']))
    y_score.append(classifierearly.predict_proba(val_df[feature_labels]).T[1])
    fpr[0],tpr[0],_ = sklearn.metrics.roc_curve(y_test[0],y_score[0])
    fpr[1],tpr[1],_ = sklearn.metrics.roc_curve(y_test[1],y_score[1])

    print subject + ' RFC has combined auc of ', (sklearn.metrics.auc(fpr[0], tpr[0]) + sklearn.metrics.auc(fpr[1], tpr[1]))/2.0
    print subject + ' has ictal auc of ' + str(sklearn.metrics.auc(fpr[0], tpr[0]))
    print subject + ' has early_ictal auc of ' + str(sklearn.metrics.auc(fpr[1], tpr[1]))
    predicted_aucRF += len(test_files)*(sklearn.metrics.auc(fpr[0], tpr[0]) + sklearn.metrics.auc(fpr[1], tpr[1]))/2.0

    y_test = []
    y_score = []
    fpr = [0,0]
    tpr = [0,0]
    # calculate ictal and early ictal ROC curves i=0 is ictal, i=1 early ictal
    classifierictal = sklearn.ensemble.AdaBoostClassifier(n_estimators=50)
    classifierictal.fit(training_df[feature_labels],training_df['ictal'])
    y_test.append(np.array(val_df['ictal']))
    y_score.append(classifierictal.predict_proba(val_df[feature_labels]).T[1])
    classifierearly = sklearn.ensemble.AdaBoostClassifier(n_estimators=50)
    classifierearly.fit(training_df[feature_labels],training_df['early_ictal'])
    y_test.append(np.array(val_df['early_ictal']))
    y_score.append(classifierearly.predict_proba(val_df[feature_labels]).T[1])
    fpr[0],tpr[0],_ = sklearn.metrics.roc_curve(y_test[0],y_score[0])
    fpr[1],tpr[1],_ = sklearn.metrics.roc_curve(y_test[1],y_score[1])

    print subject + ' ABC has combined auc of ', (sklearn.metrics.auc(fpr[0], tpr[0]) + sklearn.metrics.auc(fpr[1], tpr[1]))/2.0
    print subject + ' has ictal auc of ' + str(sklearn.metrics.auc(fpr[0], tpr[0]))
    print subject + ' has early_ictal auc of ' + str(sklearn.metrics.auc(fpr[1], tpr[1]))
    predicted_aucAB += len(test_files)*(sklearn.metrics.auc(fpr[0], tpr[0]) + sklearn.metrics.auc(fpr[1], tpr[1]))/2.0

    y_test = []
    y_score = []
    fpr = [0,0]
    tpr = [0,0]
    # calculate ictal and early ictal ROC curves i=0 is ictal, i=1 early ictal
    classifierictal = sklearn.ensemble.GradientBoostingClassifier(n_estimators=100)
    classifierictal.fit(training_df[feature_labels],training_df['ictal'])
    y_test.append(np.array(val_df['ictal']))
    y_score.append(classifierictal.predict_proba(val_df[feature_labels]).T[1])
    classifierearly = sklearn.ensemble.GradientBoostingClassifier(n_estimators=100)
    classifierearly.fit(training_df[feature_labels],training_df['early_ictal'])
    y_test.append(np.array(val_df['early_ictal']))
    y_score.append(classifierearly.predict_proba(val_df[feature_labels]).T[1])
    fpr[0],tpr[0],_ = sklearn.metrics.roc_curve(y_test[0],y_score[0])
    fpr[1],tpr[1],_ = sklearn.metrics.roc_curve(y_test[1],y_score[1])

    print subject + ' GBC has combined auc of ', (sklearn.metrics.auc(fpr[0], tpr[0]) + sklearn.metrics.auc(fpr[1], tpr[1]))/2.0
    print subject + ' has ictal auc of ' + str(sklearn.metrics.auc(fpr[0], tpr[0]))
    print subject + ' has early_ictal auc of ' + str(sklearn.metrics.auc(fpr[1], tpr[1]))
    print '\n'
    predicted_aucGB += len(test_files)*(sklearn.metrics.auc(fpr[0], tpr[0]) + sklearn.metrics.auc(fpr[1], tpr[1]))/2.0

print 'predicted RF auc: ', predicted_aucRF/32915
print 'predicted AB auc: ', predicted_aucAB/32915
print 'predicted GB auc: ', predicted_aucGB/32915
