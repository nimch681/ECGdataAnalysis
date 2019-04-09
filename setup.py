from codes.python import load_database,ECG_denoising
from codes.python import QRS_detector
import numpy as np
from scipy import signal
from scipy.signal import savgol_filter
import operator
from numpy import array
import sys
import csv
import os
import matplotlib.pyplot as plt
import wfdb
from wfdb import processing, plot
from codes.python import heartbeat_segmentation as shs
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.preprocessing import normalize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import pywt
from biosppy.signals import ecg
from sklearn import metrics
#import waipy
import operator
from codes.python import ecg_waveform_extractor as waveform
import time as system_time
from scipy import stats
import warnings
import termcolor as colored
from math import*

 

 



mit100 = DB2.patient_records[0]
t_ecpu = np.asarray(mit100.annotated_t_waves_pos[0:7],dtype=int)
wfdb.plot_items(signal = mit100.filtered_MLII[0:2000], ann_samp = [t_ecpu])


pre_rr_interval, post_rr_interval = pre_pos_rr_interval(mit100)

rr_ten, rr_fifty, rr_all = rr_average_by_sample(pre_rr_interval, fifty = True)

columns = len(mit100.segmented_beat_time[0]) + len(mit100.segmented_beat_1[0])
rows = 0
for patient in DB2.patient_records:
        rows += len(patient.segmented_beat_time)


DBn2 = np.zeros((rows,columns),dtype=object)
yn2 = np.zeros((rows,1), dtype=object)

row_count = 0

for patient in DB2.patient_records:
        
        for i in range(0,len(patient.segmented_beat_time)):
               
                row = list()
                row.extend(patient.segmented_beat_time[i])
                #time_lens.append(len(patient.segmented_beat_time[i]))
                row.extend(patient.segmented_beat_1[i])
                #beats_lens.append(len(patient.segmented_beat_1[i]))
                #if (len(patient.segmented_beat_1[i]) == 347):
                        #print(patient.filename)
                        #print(i)
                        #mit207 = patient
                
                yn2[row_count] = patient.segmented_class_ID[i]
                DBn2[row_count,0:columns] = row

                #print(DBn1[row_count])
                row_count += 1
















