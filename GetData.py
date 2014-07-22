# To be run from the parent data directory, one level above the clip files.
# Fetches the data from the specified range of clips and returns as
# a normal list, where each element is a three member list containing
# a 0 or 1 (interictal, ictal), another 0 or 1 (1 if in first 15 seconds),
# then finally a pandas dataframe containing the data.

def getData(subject, typ, start, end):

	import pandas as pd
	import scipy.io as io
	from subprocess import check_output
	from sys import exit

	data = []

	if not (check_output('find . -name ' + subject, shell=True)):
		exit('The subject specified - ' + str(subject) + ' - was not found.')

	if not (typ == 'test' or typ == 'ictal' or typ == 'interictal'):
		exit('The type specified - ' + str(typ) + ' - was not \'ictal\', \'interictal\', or \'test\'.')

	startfile = './' + subject + '/' + subject + '_' + typ + '_segment_' + str(start).zfill(4) + '.mat'
	endfile = './' + subject + '/' + subject + '_' + typ + '_segment_' + str(end).zfill(4) + '.mat'
	if not (check_output('find . -wholename ' + startfile, shell=True)):
		exit('The starting clip value specified - ' + str(start) + ' - does not exist within the data.')
	if not (check_output('find . -wholename ' + endfile, shell=True)):
		exit('The starting clip value specified - ' + str(end) + ' - does not exist within the data.')
	if not (range(int(start), int(end))):
		exit('The range specified - ' + str(start) + ' to ' + str(end) + ' - is not valid.')
	
	for x in range(int(start), int(end)+1):
		query = subject + '/' + subject + '_' + typ + '_segment_' + str(x).zfill(4) + '.mat'
		temp = io.loadmat(query)
		if (typ == 'ictal'):
			latency = temp['latency'][0]
			if (int(latency) < 15):
				temp = [1, 1, pd.DataFrame(temp['data']).T]
			else:
				temp = [1, 0, pd.DataFrame(temp['data']).T]
		else:
			temp = [0, 0, pd.DataFrame(temp['data']).T]
		
		data.append(temp)
	
	return data
