def getTraining():

	# Import needed modules.
	import pandas as pd
	import scipy.io as io
	from subprocess import check_output
	from GetData import getData
	from random import choice

	dirs = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Patient_1', 'Patient_2', 'Patient_3', 'Patient_4', 'Patient_5', 'Patient_6', 'Patient_7', 'Patient_8']

	data = {}

	for x in dirs:
		print 'Getting ' + x + '...'
		interfiles = check_output('find ' + x + ' -name *_interictal*', shell=True)
		interfiles = interfiles.split('\n')
		interfiles.pop()

		ictfiles = check_output('find ' + x + ' -name *_ictal*', shell=True)
		ictfiles = ictfiles.split('\n')
		ictfiles.pop()

		tempinter = getData(x, 'interictal', 1, len(interfiles))
		tempict = getData(x, 'ictal', 1, len(ictfiles))
		tempict2 = []

		count = 0
		ictstart = 0
		ictbounds = []
		for y in tempict:
			if(y[1] is 1 and ictstart == 0):
				print '1'
				ictbounds.append([count,0])
				ictstart = 1
			elif(y[1] is 0):
				print '2'
				ictstart = 0
			count += 1
			print 'ictstart = ' + str(ictstart) + ' ictbounds = ' + str(len(ictbounds)) + '  count = ' + str(count)
		for y in range(len(ictbounds)-1):
			ictbounds[y][1] = ictbounds[y+1][0]
		ictbounds[len(ictbounds)-1][1] = len(tempict)
		for y in ictbounds:
			tempict2.append(tempict[y[0]:y[1]])

		data[x] = [tempinter, tempict2]

	reserve = len(dirs)/3
	reserved = {} # Holds the reserved set. Not sure what to do with these...

	for x in range(reserve):
		temp = choice(data.keys())
		reserved[temp] = data[temp]
		data.pop(temp)

	return data
