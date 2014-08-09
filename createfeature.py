from glob import glob
import pandas as pd
from ecog_tools import *
import sys


#### change this to true only for the first run!!!
# otherwise you will delete all the functions you uploaded to the csv files!
create_new_files = False

if create_new_files == True:
### create Patient csv files
    for i in range(1,9):
        subject = 'Patient_%i'%(i)
        search_string = '../%s/*.mat' %(subject)
        filenames = glob(search_string)
        feature_dataframe = pd.DataFrame(index = filenames)
        feature_dataframe.to_csv('%s_features'%(subject))
        temp_df = pd.DataFrame()
        temp_df.to_csv('%s_functions'%(subject))
### create Dog csv files
    for i in range(1,5):
        subject = 'Dog_%i'%(i)
        search_string = '../%s/*.mat' %(subject)
        filenames = glob(search_string)
        feature_dataframe = pd.DataFrame(index = filenames)
        feature_dataframe.to_csv('%s_features'%(subject))
        temp_df = pd.DataFrame()
        temp_df.to_csv('%s_functions'%(subject))

#creates a list of subjects
subjects = []
for i in range(1,9):
    subject = 'Patient_%i'%(i)
    subjects.append(subject)
for i in range(1,5):
    subject = 'Dog_%i'%(i)
    subjects.append(subject)


#runs over the list of subjects!
#### this is where you need to add functions basically just change the function
#    name in add_to_featuredb with your function!
#    once you've ran this program once you don't need to keep adding the functions again.  so comment them out or delete them!
for subject in subjects:
    print subject
    sys.stdout.flush()
    #    add_to_featuredb(corr,'%s_features'%(subject),'%s_functions'%(subject))
    #add_to_featuredb(var,'%s_features'%(subject),'%s_functions'%(subject))
    add_to_featuredb(ictal,'%s_features'%(subject),'%s_functions'%(subject))
    add_to_featuredb(early_ictal,'%s_features'%(subject),'%s_functions'%(subject))
    #add_to_featuredb(deltapower,'%s_features'%(subject),'%s_functions'%(subject))
    #add_to_featuredb(thetapower,'%s_features'%(subject),'%s_functions'%(subject))
    #add_to_featuredb(alphapower,'%s_features'%(subject),'%s_functions'%(subject))
    #add_to_featuredb(betapower,'%s_features'%(subject),'%s_functions'%(subject))
    #add_to_featuredb(lgammapower,'%s_features'%(subject),'%s_functions'%(subject))
    #add_to_featuredb(hgammapower,'%s_features'%(subject),'%s_functions'%(subject))