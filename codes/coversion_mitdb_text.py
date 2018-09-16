#Script from https://github.com/mondejar/WFDB-utils-and-others/blob/master/convert_wfdb_data_2_csv.py
# this script convert the full dataset mitdb (data and annotatiosn) to text files

from os import listdir, mkdir, system
from os.path import isfile, isdir, join, exists
import os
#dir = 'database/mitdb/'
#Create folder
#dir_out = dir + 'csv/'
#if not exists(dir_out):
	#mkdir(dir_out)

#records = [f for f in listdir(dir) if isfile(join(dir, f)) if(f.find('.dat') != -1)]
#print records

for r in range(100,110):

	command = 'rdsamp -r mitdb/' +str(r)+ ' -c -H -f 0 -t 60 -v -ps >' + str(r) + '.csv'
	print(command)
	system(command)

	command_annotations = 'rdann -r mitdb/' +str(r)+' -f 0 -a atr -v >' +str(r)+ '.ann'
	print(command_annotations)
	system(command_annotations)