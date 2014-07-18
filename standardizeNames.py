# Run in parent directory containing each subject's folder, i.e.
# */Volumes/Seagate/seizure_detection/competition_data/clips
# Will go through all the files and rename them so that the numeric
# identifier is a 4 digit number (ensuring they are loaded in the
# same order.
# e.g. 'Dog_1_ictal_segment_1.mat' --> 'Dog_1_ictal_segment_0001.mat'

# Import requisite modules.
from subprocess import check_output, call
import re

# Fetch the names of all the folders in the current directory
# and process into a list.
folders = check_output('ls -d */', shell = True)
folders = folders.split('\n')
folders.pop()

# Regexp for finding the numeric indentifier for each .mat file.
renumber = re.compile('_(\d{1,4})\.mat')

# Iterate through the folders, converting all the names.
for f in folders:
	print 'Starting ' + str(f)

	# Get the list of file names in the current folder.
	files = check_output(['ls', str(f)])
	files = files.split('\n')
	files.pop()
	totalfiles = len(files)
	count = 0
	
	# If the file has a numeric identifier less than 4 digits,
	# zfill it such that it has 4 digits so that they will
	# load in the proper order.
	for n in files:
		oldnum = re.search(renumber, n).group(1)
		if(len(oldnum) < 4):
			newnum = str(oldnum).zfill(4)
			newname = str(f) + '/' + re.sub(oldnum + '.mat', newnum + '.mat', n)
			oldname = str(f) + '/' + str(n)
			call(['mv', oldname, newname])
		count += 1
		if(count % 100 == 0):
			print str(count) + '/' + str(totalfiles)

	print 'Finished ' + str(f) + ' ' + str(count) + '/' + str(totalfiles)
