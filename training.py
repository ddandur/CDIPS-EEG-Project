from ecog_tools import *
import pandas as pd
import numpy as np
import sklearn
import sklearn.ensemble
import scipy.stats
from random import shuffle

total_test_df = pd.DataFrame()

subjects = []
for i in range(1,9):
    subject = 'Patient_%i'%(i)
    subjects.append(subject)
for i in range(1,5):
    subject = 'Dog_%i'%(i)
    subjects.append(subject)

########

predicted_auc = 0


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
    v_percent = 0.20

    val_set = early_ictal_files[:int(v_percent*len(early_ictal_files))]
    train_set = early_ictal_files[int(v_percent*len(early_ictal_files)):]

    val_set += ictal_files[:int(v_percent*len(ictal_files))]
    train_set += ictal_files[int(v_percent*len(ictal_files)):]

    val_set += inter_ictal_files[:int(v_percent*len(inter_ictal_files))]
    train_set += inter_ictal_files[int(v_percent*len(inter_ictal_files)):]

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
    for i in range(2):
        classifier = sklearn.ensemble.RandomForestClassifier(n_estimators=100)
        classifier.fit(training_df[feature_labels],training_df[classifier_labels])
        y_test.append(np.array(val_df[classifier_labels[i]]))
        y_score.append(classifier.predict_proba(val_df[feature_labels])[i].T[1])
        fpr[i],tpr[i],_ = sklearn.metrics.roc_curve(y_test[i],y_score[i])

    print subject + ' has combined auc of ', (sklearn.metrics.auc(fpr[0], tpr[0]) + sklearn.metrics.auc(fpr[1], tpr[1]))/2.0
    predicted_auc += len(test_files)*(sklearn.metrics.auc(fpr[0], tpr[0]) + sklearn.metrics.auc(fpr[1], tpr[1]))/2.0

    file_format_dict = {}
    for test_file in test_files:
        search_string = '%s_'%(subject)
        file_format_dict[test_file] = test_file[test_file.find(search_string):]
    test_df = test_df.apply(lambda x: x.fillna(x.mean()),axis=0)
    test_df['seizure'] = classifier.predict_proba(test_df[feature_labels])[0].T[1]
    test_df['early'] = classifier.predict_proba(test_df[feature_labels])[1].T[1]
    test_df = test_df.drop([i for i in test_df.columns if i != 'seizure' and i != 'early'],axis =1)
    test_df = test_df.rename(index=file_format_dict)
    total_test_df = total_test_df.combine_first(test_df)
total_test_df.index.name = 'clip'
total_test_df.to_csv('submission.csv')
print 'len: ', len(total_test_df), ' which should be 32915'
print 'predicted auc: ', predicted_auc/len(total_test_df)
