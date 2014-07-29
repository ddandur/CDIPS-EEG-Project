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

# Power Function: As a first pass, each power function calculates the power spectrum 
# for all electrodes, averages the spectrum across electrodes, 
# and finally bins the data into the selected frequency range.. This way the output 
# from each power function call is still a single real value. 
 
# The lowest sampling frequency is 400 Hz, which means the highest frequency 
# we should ask about should be 200 Hz. Thus lower_hz should be zero or above 
# and upper_hz should be 200 and below, with lower_hz < upperhz. 
# The input variables lower_hz and upper_hz should be integers. 

def power(filename, lower_hz, upper_hz):
    sample = scipy.io.loadmat(filename)
#   num_el = number of electrodes     
    num_el = len(sample['data'])
    
# Calculate the real fast fourier transform on each electrode; 
# the way it's done here is to take modulus of each positive 
# complex frequency. The result pwr_array is still an array, a list with 
# num_el values where each element is the corresponding electrode's 
# values for the FFT squared, each value corresponding to power 
# at frequency 0 Hz, 1 Hz, etc up to the Nyquist frequency 
    freq_array = abs(np.fft.rfft(sample['data']))
    pwr_array = freq_array**2
    
# Now average the power across all the electrodes; 
# ave_pwr is a single array of average power values at each frequency 
# among the electrodes 
    sum_pwr = np.sum(pwr_array, axis = 0)
    ave_pwr = np.sum/float(num_el)
    
# Now pick out the power values from the desired frequency range 
# and sum them. A nice feature for us is that the list index of the elements 
# in the ave_pwr list is also the frequency corresponding to that element in the list. 
    selected_part_of_list = ave_pwr[int(lower_hz):int(upper_hz + 1)]
    return np.sum(selected_part_of_list)
        

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
