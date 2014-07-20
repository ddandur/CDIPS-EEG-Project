# Run in the directory containing the 12 subject folders.
from subprocess import check_output
from scipy.io import loadmat
import re


REtyp = re.compile('\d_(\w+)_segment')
REnum = re.compile('segment_(\d+)\.mat')

folderlist = check_output('ls -d */', shell=True).split('\n')
folderlist.pop()
filelist = []

for x in folderlist:
	y = check_output('ls ' + x, shell=True).split('\n')
	y.pop()
	filelist.append(y)

for x in filelist:
	freq = loadmat(folderlist[filelist.index(x)] + x[0])
	freq = freq['freq'][0]
	ictalcount = 0
	interictalcount = 0
	testcount = 0
	seizures = []
	for y in x:
		typ = re.search(REtyp, y).group(1)
		if (typ == 'ictal'):
			ictalcount += 1
			icdata = loadmat(folderlist[filelist.index(x)] + y)
			latency = icdata['latency'][0]
			if (latency == 0):
				seizures.append([y])
			else:
				seizures[len(seizures)-1].append(y)
		elif (typ == 'interictal'):
			interictalcount += 1
		else:
			testcount += 1

	print folderlist[filelist.index(x)] + ':'
	print '\tFrequency: ' + str(freq)
	print '\tInterictal files: ' + str(interictalcount)
	print '\tTest files: ' + str(testcount)
	print '\tIctal files: ' + str(ictalcount) + ' (' + str(len(seizures)) + ' seizures)'
	for y in seizures:
		print '\t\t' + str(re.search(REnum, y[0]).group(1)) + ' - ' + str(re.search(REnum, y[len(y)-1]).group(1)) + ': ' + str(len(y)) + ' seconds'
	print ''
	
