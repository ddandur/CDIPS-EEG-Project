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
def is_mat(filename):
    return filename.find('.mat') != -1
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
        ######## This is under dispute whether < 15 or < 16
        return tempfile['latency'][0] < 16
    else:
        raise TypeError('file must be ictal')

#number value functions that act on the file names
#input: a string of the .mat filename
def mean_var(filename):
    sample = scipy.io.loadmat(filename)
    ecog_rec = pd.DataFrame(sample['data'])
    return [[mean_var.__name__,np.mean(ecog_rec.T.var())]]
def mean_corr(filename):
    sample = scipy.io.loadmat(filename)
    n_e = len(sample['data'])
    corr_matrix = np.corrcoef(sample['data'])
    np.fill_diagonal(corr_matrix,0)
    return [[mean_corr.__name__,sum(sum(corr_matrix))/(n_e**2-n_e)]]
def ictal(filename):
    if is_test(filename):
        return [[ictal.__name__,np.nan]]
    return [[ictal.__name__,int(is_ictal(filename))]]
def early_ictal(filename):
    if is_test(filename):
        return [[early_ictal.__name__,np.nan]]
    elif is_ictal(filename):
        return [[early_ictal.__name__,int(is_early_ictal(filename))]]
    else:
        return [[early_ictal.__name__,0]]

#array value functions that act on the file names
#input: a string of the .mat filename
def var(filename):
    sample = scipy.io.loadmat(filename)
    ecog_rec = pd.DataFrame(sample['data'])
    var_values = list(ecog_rec.T.var())
    output_list = []
    for i in range(len(sample['data'])):
        if var_values[i] == 0:
            var_values[i] = np.nan
        output_list.append(['var_%i'%(i),var_values[i]])
    return output_list

def corr(filename):
    sample = scipy.io.loadmat(filename)
    ecog_rec = pd.DataFrame(sample['data'])
    var_values = list(ecog_rec.T.var())

    corr_matrix = np.corrcoef(sample['data'])
    output_list = []
    for i in range(len(sample['data'])):
        for j in range(i+1,len(sample['data'])):
            if var_values[i]==0 or var_values[j]==0:
                corr_matrix[i,j] = np.nan
            output_list.append(['corr_%i_%i'%(i,j),corr_matrix[i,j]])
    return output_list




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
    ave_pwr = sum_pwr/float(num_el)
    
    # Now pick out the power values from the desired frequency range
    # and sum them. A nice feature for us is that the list index of the elements
    # in the ave_pwr list is also the frequency corresponding to that element in the list.
    selected_part_of_list = ave_pwr[int(lower_hz):int(upper_hz + 1)]
    return np.sum(selected_part_of_list)


def deltapower(filename):
    return [[deltapower.__name__,power(filename,0.1,4)]]
def thetapower(filename):
    return [[thetapower.__name__,power(filename,4,8)]]
def alphapower(filename):
    return [[alphapower.__name__,power(filename,8,12)]]
def betapower(filename):
    return [[betapower.__name__,power(filename,12,30)]]
def lgammapower(filename):
    return [[lgammapower.__name__,power(filename,30,70)]]
def hgammapower(filename):
    return [[hgammapower.__name__,power(filename,70,180)]]


#####
#####
#####


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

#creates a dataframe with features generated from one meathod:
#input: a function that operates on locations that return one value
#       and a list of datafiles
#output: corresponding pandas dataframe
def labeled_dataframe(function,datafiles):


    dataframe = pd.DataFrame.from_dict(labeled_data([function],datafiles),orient='index')
    dataframe.columns = [function.__name__]
    return dataframe

#overides feature values in csv_file with values in dataframe
#input: function that need to be added to feature database
#       and csv_feature_file which is the string of the csv file location
#       and csv_function_file which is the string of the csv file location
#output: no return value, modifies the original csv files
def add_to_featuredb(function, csv_feature_file, csv_function_file):
    feature_dataframe = pd.DataFrame.from_csv(csv_feature_file)
    filenames = list(feature_dataframe.index)
    if len(filenames) != len(set(filenames)):
        raise TypeError('entries must be uniquely named')
    new_feature_df = pd.DataFrame()
    for filename in filenames:
        if not is_mat(filename):
            raise TypeError('files must be \'.mat\' files')
        column_names = []
        values = []
        for output in function(filename):
            column_names.append(output[0])
            values.append(output[1])
        value_dataframe = pd.DataFrame([values], index = [filename], columns = column_names)
        new_feature_df = new_feature_df.append(value_dataframe)
    column_names = []
    for output in function(filenames[0]):
        column_names.append(output[0])
    function_dataframe = pd.DataFrame.from_csv(csv_function_file)
    new_function_df = pd.DataFrame([[column_names]],index = [function.__name__], columns = ['column_names'])

    feature_dataframe = new_feature_df.combine_first(feature_dataframe)
    function_dataframe = new_function_df.combine_first(function_dataframe)
    feature_dataframe.to_csv(csv_feature_file)
    function_dataframe.to_csv(csv_function_file)

#creates a dataframe with features only generated from the list of functions
#input: a list of functions, and the csv_file location as a string
#output: corresponding pandas dataframe
def extract_features(functions,csv_feature_file, csv_function_file):
    function_dataframe = pd.DataFrame.from_csv(csv_function_file)
    feature_dataframe = pd.DataFrame.from_csv(csv_feature_file)
    column_names = []
    for function in functions:
        if function.__name__ not in function_dataframe.index:
            raise TypeError('functon ' + function.__name__ + ' has not been evaluated on the feature database')
        for feature in eval(function_dataframe.get_value(function.__name__,'column_names')):
            column_names.append(feature)
    return feature_dataframe.loc[feature_dataframe.index,column_names]



