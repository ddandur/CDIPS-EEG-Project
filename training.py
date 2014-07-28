# Training program.

from ecog_tools import *
from random import shuffle
from glob import glob

# Create a list of all .mat file names.
eeg_mat_fnames = glob('*/*.mat')

# Seperate files by type.
test_files = [file for file in eeg_mat_fnames if is_test(file)]

# Shuffle files to seperate for training and validation.
# This will ulitmately cause a portion to be kept for validation,
# BUT this portion is different each time, I'm not sure if that
# is good or bad?
ictal_files = [file for file in eeg_mat_fnames if is_ictal(file)]
early_ictal_files = [file for file in ictal_files if is_early_ictal(file)]
ictal_files = [file for file in ictal_files if not is_early_ictal(file)]
shuffle(ictal_files)
shuffle(early_ictal_files)
inter_ictal_files =[file for file in eeg_mat_fnames if is_inter_ictal(file)]
shuffle(inter_ictal_files)


# Break up into validation and training set.
# v_percent of the data is withheld for validation.
v_percent = .25

val_set = early_ictal_files[:int(v_percent*len(early_ictal_files))]
training_set = early_ictal_files[int(v_percent*len(early_ictal_files)):]

val_set += ictal_files[:int(v_percent*len(ictal_files))]
training_set += ictal_files[int(v_percent*len(ictal_files)):]

val_set += inter_ictal_files[:int(v_percent*len(inter_ictal_files))]
training_set += inter_ictal_files[int(v_percent*len(inter_ictal_files)):]

shuffle(val_set)
shuffle(training_set)


# This is where we can add or remove labels and features.
# All functions act on strings with .mat filenames that can be
# called in that directory.
# Suggest adding functions with clear names in 'ecog_tools.py'.
column_functions = [ictal , early_ictal, mean_corr, mean_var]
feature_functions = [mean_var,mean_corr]

column_labels = [i.__name__ for i in column_functions]
feature_labels = [i.__name__ for i in feature_functions]

# Create training data as a dictionary where the keys are file names
# and the values are a list of values corresponding to the labels
# and features.
training_data = labeled_data(column_functions, training_set)



########
# Use different algorithms to train on training_data








