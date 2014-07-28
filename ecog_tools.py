#collection of functions to handle Kaggle 2014 ECOG data
import scipy
import scipy.io
import numpy as np
import pandas as pd

#intended to transform data into a usable format to train on
#input: a list of functions and a list of data
#output: a dictionary with keys being the entries in data
#note: elements of data must be unique
def labeled_data(functions,data):
    dictdata = {}
    for sample in data:
        temp_list = []
        for f in functions:
            temp_list.append(f(sample))
        if sample in dictdata:
            raise TypeError('values in data input not unique')
        dictdata[sample] = temp_list
    return dictdata

#boolean functions that act on the file names
#input: a string of the .mat filename
def is_dog(filename):
    return filename.find('Dog') != -1
def is_human(filename):
    return filename.find('Patient') != -1
def is_inter_ictal(filename):
    return filename.find('inter') != -1
def is_test(filename):
    return filename.find('test') != -1
def is_ictal(filename):
    return filename.find('ictal') != -1 and not (is_inter_ictal(filename) or is_test(filename))
def is_early_ictal(filename):
    if is_ictal(filename):
        tempfile = scipy.io.loadmat(filename)
        return tempfile['latency'][0] < 15
    else:
        raise TypeError('file must be ictal')

#number value functions that act on the file names
#input: a string of the .mat filename
def mean_var(filename):
    sample = scipy.io.loadmat(filename)
    ecog_rec = pd.DataFrame(sample['data'])
    return np.mean(ecog_rec.T.var())
def mean_corr(filename):
    sample = scipy.io.loadmat(filename)
    n_e = len(sample['data'])
    corr_matrix = np.corrcoef(sample['data'])
    np.fill_diagonal(corr_matrix,0)
    return sum(sum(corr_matrix))/(n_e**2-n_e)
def ictal(filename):
    return int(is_ictal(filename))
def early_ictal(filename):
    if is_ictal(filename):
        return int(is_early_ictal(filename))
    else:
        return 0
