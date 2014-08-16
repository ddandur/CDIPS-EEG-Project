from ecog_tools import *
import pandas as pd
import numpy as np
import sklearn
import sklearn.ensemble
from sklearn import svm
import scipy.stats
import random

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
#subjects = [subjects[11]]

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

    ictal_files = [file for file in eeg_mat_fnames if is_ictal(file)]
    inter_ictal_files =[file for file in eeg_mat_fnames if is_inter_ictal(file)]
    
    #create val and train set
    ictal_segments = []
    val_set = []
    train_set = []
    # goes through ictal files
    begin = 1
    for i in range(2,len(ictal_files)+1):
        tempfile = scipy.io.loadmat('../%s/%s_ictal_segment_%i.mat'%(subject,subject,i))
        if tempfile['latency'][0] == 0:
            ictal_segments.append([begin,i-1])
            begin = i
    ictal_segments.append([begin,len(ictal_files)])
    print ictal_segments
    if len(ictal_segments) < 6:
        n_train_seizures = 1
    elif len(ictal_segments) > 5:
        n_train_seizures = 2
    random.shuffle(ictal_segments)
    for j in range(n_train_seizures):
        temp_segment = ictal_segments.pop()
        for i in range(temp_segment[0],temp_segment[1]+1):
            val_set.append('../%s/%s_ictal_segment_%i.mat'%(subject,subject,i))
    for ictal_segment in ictal_segments:
        for i in range(ictal_segment[0],ictal_segment[1]+1):
            train_set.append('../%s/%s_ictal_segment_%i.mat'%(subject,subject,i))
    print len(ictal_files),len(val_set)+len(train_set) ,len(val_set),len(train_set)
    #goes through inter_ictal files
    #makes sure the validation set is roughly in proportion to the training set
    percent_val = float(len(val_set))/float(len(ictal_files))
    print percent_val
    print len(inter_ictal_files)
    print len(inter_ictal_files)+int(-percent_val*len(inter_ictal_files))
    print random.randint(1,1000)
    begin = random.randint(1,len(inter_ictal_files)+int(-percent_val*len(inter_ictal_files)))
    end = begin + int(percent_val*len(inter_ictal_files))
    for i in range(1,len(inter_ictal_files)+1):
        if i < begin or i > end:
            train_set.append('../%s/%s_interictal_segment_%i.mat'%(subject,subject,i))
        else:
            val_set.append('../%s/%s_interictal_segment_%i.mat'%(subject,subject,i))

    random.shuffle(val_set)
    random.shuffle(train_set)
    print len(val_set)
    print len(train_set)
    print len(val_set)+len(train_set)
    print len(inter_ictal_files)+len(ictal_files)
    '''
    ### this was the old validation meathod
    v_set = val_set
    t_set = train_set
    t_new_set = val_set +train_set

    # Shuffle files to seperate for training and validation.
    ictal_files = [file for file in eeg_mat_fnames if is_ictal(file)]
    early_ictal_files = [file for file in ictal_files if is_early_ictal(file)]
    ictal_files = [file for file in ictal_files if not is_early_ictal(file)]
    random.shuffle(ictal_files)
    random.shuffle(early_ictal_files)
    inter_ictal_files =[file for file in eeg_mat_fnames if is_inter_ictal(file)]
    random.shuffle(inter_ictal_files)

    # Break up into validation and training set.
    # v_percent of the data is withheld for validation.
    v_percent = 0.20

    val_set = early_ictal_files[:int(v_percent*len(early_ictal_files))]
    train_set = early_ictal_files[int(v_percent*len(early_ictal_files)):]

    val_set += ictal_files[:int(v_percent*len(ictal_files))]
    train_set += ictal_files[int(v_percent*len(ictal_files)):]

    val_set += inter_ictal_files[:int(v_percent*len(inter_ictal_files))]
    train_set += inter_ictal_files[int(v_percent*len(inter_ictal_files)):]

    random.shuffle(val_set)
    random.shuffle(train_set)

    t_set = val_set + train_set
    print t_set[0],t_new_set[0]
    print '    ##########!'
    for i in t_new_set:
        if i not in t_set:
            print i
    print '    ###########'
    for i in t_set:
        if i not in t_new_set:
            print i

    print t_set == t_new_set
    '''
#    train_set += val_set
#    val_set = train_set
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
        #classifier = sklearn.ensemble.RandomForestClassifier(n_estimators=100)
        classifier = sklearn.ensemble.AdaBoostClassifier()
        classifier.fit(np.array(training_df[feature_labels]),np.array(training_df[classifier_labels[i]]))
        print 'testing'
        y_test.append(np.array(val_df[classifier_labels[i]]))
        print y_score.append(classifier.predict_proba(val_df[feature_labels]).T[1])
        fpr[i],tpr[i],_ = sklearn.metrics.roc_curve(y_test[i],y_score[i])

    print subject + ' has combined auc of ', (sklearn.metrics.auc(fpr[0], tpr[0]) + sklearn.metrics.auc(fpr[1], tpr[1]))/2.0
    predicted_auc += len(test_files)*(sklearn.metrics.auc(fpr[0], tpr[0]) + sklearn.metrics.auc(fpr[1], tpr[1]))/2.0

    file_format_dict = {}
    for test_file in test_files:
        search_string = '%s_'%(subject)
        file_format_dict[test_file] = test_file[test_file.find(search_string):]
    #something better needs to go here:
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
'''
'''