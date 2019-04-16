import numpy as np 
import math
import matplotlib.pyplot as plt
from numpy import array
from codes.python import QRS_detector
import sys
import csv
import os
import operator
from numpy import array
import sys
import csv
import os
import matplotlib.pyplot as plt
import wfdb
from wfdb import processing, plot



mit100 = load_database.load_patient_record("mitdb", "100")
mit100.set_segmented_beats_r_pos(winL=100,winR=200)
mit100.set_r_properties_MLII()
mit100.set_Q_S_points_MLII()
mit100.set_P_T_points_MLII()
mit100.set_rr_intervals()
mit100.set_intervals_and_averages()





QRS_properties, P_Q_properties, P_Q_neg, P_R_properties, P_R_neg, S_T_properties, S_T_neg,R_T_properties, R_T_neg, P_T_properties,neg_P_T, P_T_neg, neg_P_T_neg =interval_and_average(mit100)

patient_list_1 = ["101","106","108","109","112","114","115","116","118","119","122","124","201","203","205","207","208","209","215","220","223","230"]
patient_list_2 = ["100","103","105","111","113","117","121","123","200","202","210","212","213","214","219","221","222","228","231","232","233","234"]
DB1 = load_database.create_ecg_database("mitdb",patient_list_1)
DB2 = load_database.create_ecg_database("mitdb",patient_list_2)
DB1.segment_beats()
DB2.segment_beats()

#####################################

mit100 = DB2.patient_records[0]



columns = (13*7) + 4 + (3*6) +7
rows = len(mit100.segmented_R_pos)



for patient in DB1.patient_records:
        rows += len(patient.segmented_beat_time)


x = np.zeros((rows,columns),dtype=object)
y = np.zeros((rows,1), dtype=object)

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

for i in range(0,len(mit100.segmented_beat_time)):
               
    row = list()
    row.extend(patient.segmented_beat_time[i])
    #time_lens.append(len(patient.segmented_beat_time[i]))
    row.extend(patient.segmented_beat_1[i])
    #beats_lens.append(len(patient.segmented_beat_1[i]))
        #if (len(patient.segmented_beat_1[i]) == 347):
            #print(patient.filename)
            #print(i)
            #mit207 = patient
                
    y[row_count] = patient.segmented_class_ID[i]
    x[row_count,0:columns] = row

    #print(DBn1[row_count])
    row_count += 1