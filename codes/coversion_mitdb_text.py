#Script from https://github.com/mondejar/WFDB-utils-and-others/blob/master/convert_wfdb_data_2_csv.py
# this script convert the full dataset mitdb (data and annotatiosn) to text files

from os import listdir, mkdir, system
from os.path import isfile, isdir, join, exists
import os
os.getcwd()

dir = '/home/nimch681/Documents/git_projects/ECGdataAnalysis/database/mitdb/'
#Create folder
dir_out = dir + 'csv/'
if not exists(dir_out):
	mkdir(dir_out)

#records = [f for f in listdir(dir) if isfile(join(dir, f)) if(f.find('.dat') != -1)]
#print records

for r in range(100,110):
	command = 'rdsamp -r mitdb/'  +r+ ' -c -H -f 0 -60 -v -ps >' + r + '.csv'
	print(command)
	system(command)

	command_annotations = 'rdann -r mitdb/'  +r+ ' -c -H -f 0 -60 -v -ps >' + r +'.text'
	print(command_annotations)
	system(command_annotations)
