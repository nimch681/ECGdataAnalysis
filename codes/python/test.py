import numpy as np 
import math
import matplotlib.pyplot as plt
from numpy import array
from codes.python import QRS_detector
import sys
import csv
import os
import operator


np100 = array([mit100.MLII])
np100T = np100.T


def load_signal( filename ):
    # Read data from file .csv 
    ecg_signal = list()
    with open(filename, 'rt') as f:
        reader = csv.reader(f)
        for row in reader:
            ecg_signal.append(float(row[0]))

    fs = ecg_signal[0]
    min_A = ecg_signal[1]
    max_A = ecg_signal[2]
    n_bits = ecg_signal[3]
    ecg_signal = ecg_signal[4:]   
    
    return ecg_signal, fs, min_A, max_A, n_bits


ecg_signal, fs, min_A, max_A, n_bits = load_signal("220.csv")

qrs_detector = QRS_detector.QRSDetectorOffline(ecg_data_raw = ecg_signal, fs = fs, verbose=True, plot_data=True, show_plot=True)

load_mitdb.display_signal(ecg_signal)


MITBIH_classes = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F']#, 'P', '/', 'f', 'u']
AAMI_classes = []
AAMI_classes.append(['N', 'L', 'R'])                    # N
AAMI_classes.append(['A', 'a', 'J', 'S', 'e', 'j'])     # SVEB 
AAMI_classes.append(['V', 'E'])                         # VEB
AAMI_classes.append(['F'])       



def find_beats(patient_record, winL, winR):
    class_ID = []
    beat = []
    R_poses = []
    #Original_R_poses = []  
   # valid_R = np.zeros((2,))
    size_RR_max = 22

    for a in patient_record.annotations:
        aS = a.split()
            
        pos = float(aS[1])
        originalPos = float(aS[1])
        classAnttd = aS[2]
        if pos > size_RR_max and pos < (len(patient_record.MLII) - size_RR_max):
            index, value = max(enumerate(patient_record.MLII[pos - size_RR_max : pos + size_RR_max]), key=operator.itemgetter(1))
            pos = (pos - size_RR_max) + index

        peak_type = 0
        #pos = pos-1
            
      #  if classAnttd in MITBIH_classes:
      #      if(pos > winL and pos < (len(patient_record.MLII) - winR)):
          #     beat.append( (patient_record.MLII[pos - winL : pos + winR], patient_record.V1[pos - winL : pos + winR]))
           #     for i in range(0,len(AAMI_classes)):
            #        if classAnttd in AAMI_classes[i]:
             #           class_AAMI = i
              #          break #exit loop
                #convert class
               # class_ID.append(class_AAMI)

                #R_poses.append(pos)
           # else:
               # valid_R = np.append(valid_R, 0)
       # else:
            #valid_R = np.append(valid_R, 0)
            
        #R_poses.append(pos)
    
        #Original_R_poses = np.append(Original_R_poses, originalPos)
    patient_record.beat = beat
    patient_record.class_ID = class_ID
    patient_record.R_pos = R_poses

split100 = mit100.MLII.split
mit100 = load_patient_record("mitdb","100")
class_ID = []
beat = []
R_poses = []
    #Original_R_poses = []  
   # valid_R = np.zeros((2,))
size_RR_max = 22
pos = 0
aS = []
winL = 90
winR = 90
i = 0
for a in mit100.annotations:
    
    aS = a.split()
    if(i == 0):
        print(aS)     
    pos = int(aS[1])
    originalPos = int(aS[1])
    classAnttd = str(aS[2])
    #originalPos = int(aS[1])
   # classAnttd = aS[2]
   # if pos > size_RR_max and pos < (len(mit100.MLII) - size_RR_max):
        #index, value = max(enumerate(mit100.MLII[pos - size_RR_max : pos + size_RR_max]), key=operator.itemgetter(1))
        #pos = (pos - size_RR_max) + index
    i = i+1

    if pos > size_RR_max and pos < (len(mit100.MLII) - size_RR_max):
            index, value = max(enumerate(mit100.MLII[pos - size_RR_max : pos + size_RR_max]), key=operator.itemgetter(1))
            pos = (pos - size_RR_max) + index
            
    peak_type = 0
        
    print("classAnttd " + classAnttd)        
    if classAnttd in MITBIH_classes:
        print("Got this class in MITBIH classAnttd " + classAnttd )
        if(pos > winL and pos < (len(mit100.MLII) - winR)):
            beat.append( (mit100.MLII[pos - winL : pos + winR], mit100.V1[pos - winL : pos + winR]))
            print("hi")
            for i in range(0,len(AAMI_classes)):
                if classAnttd in AAMI_classes[i]:
                    class_AAMI = i
                    break #exit loop
                #convert class
            class_ID.append(class_AAMI)

            R_poses.append(pos)
           # else:
               # valid_R = np.append(valid_R, 0)
       # else:
            #valid_R = np.append(valid_R, 0)
            
        #R_poses.append(pos)
    
        #Original_R_poses = np.append(Original_R_poses, originalPos)
mit100.beat = beat
mit100.class_ID = class_ID
mit100.R_pos = R_poses
   # peak_type = 0




beat, class_ID, R_poses = segment_beat(mit100.filtered_V1, mit100.annotations, 90, 90)