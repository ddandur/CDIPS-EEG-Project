# 2014-07-18 First iteration of arbitrary 16 electrode plotting function.
# User specifies subject (e.g. Dog_1), type of mat file (e.g. ictal, 
# interictal, or test), and the start (e.g. 1) and stop (e.g. 2) time points
# on the command line, and the function returns a graph.
# Open to extension and modification to suit given needs!

# Import requisit modules.
import scipy.io as io
import matplotlib.pyplot as plt
from sys import argv
import re
from subprocess import call, check_output

# Command line arguments (all imported as strings):
# [<plotl.py>, <subject folder>, <ictal/interictal/test>, <startid>, <endid>]
if (len(argv) != 5):
	print 'Improper number of arguments provided.'
	print 'Expected: python plot1.py <SUBJECT FOLDER> <TYPE> <START> <END> e.g.:'
	print '<SUBJECT FOLDER> = Dog_1'
	print '<TYPE> = ictal'
	print '<START> = 1'
	print '<END> = 20'

SUBJECT = argv[1]
TYPE = argv[2]
START = str(argv[3]).zfill(4)
END = str(argv[4]).zfill(4)

# Print summary of operation for user to see.
print ('Plotting ' + TYPE + ' for ' + SUBJECT + ' from ' + START + ' to ' + END + ':')

# Construct the list of source files to open.
sources = []
for x in range(int(START), int(END)+1):
	sources.append(SUBJECT + '/' + SUBJECT + '_' + TYPE + '_segment_' + str(x).zfill(4) + '.mat')

# Load in the data from the sources for each electrode.
electrodes = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
for s in sources:
	temp = io.loadmat(s)
	for x in range(0,16):
		electrodes[x].extend(temp['data'][x])

count = 0
sp = []
for n in range(1, len(electrodes)+1):
	sp.append('e' + str(n))

figure, sp = plt.subplots(len(electrodes), sharex = True, sharey = True)
sp[0].set_title(SUBJECT + ' ' + TYPE + ' ' + str(START) + ' to ' + str(END))
for e in electrodes:
	sp[count].plot(e)
	count += 1

plt.setp([a.get_yticklabels() for a in figure.axes[0:len(electrodes)]], visible=False)
figure.subplots_adjust(hspace=0)
plt.show()
