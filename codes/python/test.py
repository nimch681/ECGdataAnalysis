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


mitdb = load_database.load_mitdb()
mit100 = mitdb.patient_records[0]
mit100.set_segmented_beats_r_pos(winL=100,winR=180)
mit100.set_r_properties_MLII()
mit100.set_Q_S_points_MLII()
mit100.set_P_T_points_MLII()
mit100.set_rr_intervals()
mit100.set_intervals_and_averages()





QRS_properties, P_Q_properties, P_Q_neg, P_R_properties, P_R_neg, S_T_properties, S_T_neg,R_T_properties, R_T_neg, P_T_properties,neg_P_T, P_T_neg, neg_P_T_neg =interval_and_average(mit100)



#####################################




columns = (13*7) + 5 + (3*6) +7
rows = len(mit100.segmented_R_pos)



#for patient in mitdb.patient_records:
        #rows += len(patient.segmented_beat_time)


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


patient = mit100
for i in range(0,len(patient.segmented_beat_time)):
               
    row = list()
    row.extend(patient.R_pos_properites["durations"][i])
    row.extend(patient.R_pos_properites["height"][i])
    row.extend(patient.R_pos_properites["amplitudes"][i])
    row.extend(patient.R_pos_properites["prominences"][i])

    row.extend(patient.Q_points_properites["durations"][i])
    row.extend(patient.Q_points_properites["height"][i])
    row.extend(patient.Q_points_properites["amplitudes"][i])
    row.extend(patient.Q_points_properites["prominences"][i])

    row.extend(patient.S_points_properites["durations"][i])
    row.extend(patient.S_points_properites["height"][i])
    row.extend(patient.S_points_properites["amplitudes"][i])
    row.extend(patient.S_points_properites["prominences"][i])

    p_durations=np.asarray(patient.P_points_properites["durations"])
    p_height=np.asarray(patient.P_points_properites["height"])
    p_amplitudes=np.asarray(patient.P_points_properites["amplitudes"])
    p_prominence = np.asarray(patient.P_points_properites["prominences"])


    row.extend(p_durations[i,0])
    row.extend(p_height[i,0])
    row.extend(p_amplitudes[i,0])
    row.extend(p_prominence[i,0])

    row.extend(p_durations[i,1])
    row.extend(p_height[i,1])
    row.extend(p_amplitudes[i,1])
    row.extend(p_prominence[i,1])

    t_durations=np.asarray(patient.T_points_properites["durations"])
    t_height=np.asarray(patient.T_points_properites["height"])
    t_amplitudes=np.asarray(patient.T_points_properites["amplitudes"])
    t_prominence = np.asarray(patient.T_points_properites["prominences"])


    row.extend(t_durations[i,0])
    row.extend(t_height[i,0])
    row.extend(t_amplitudes[i,0])
    row.extend(t_prominence[i,0])

    row.extend(t_durations[i,1])
    row.extend(t_height[i,1])
    row.extend(t_amplitudes[i,1])
    row.extend(t_prominence[i,1])

    row.extend(patient.rr_interval["pre"][i])
    row.extend(patient.rr_interval["post"][i])
    row.extend(patient.rr_interval["average_ten"][i])
    row.extend(patient.rr_interval["average_fifty"][i])
    row.extend(patient.rr_interval["average_all"][i])

    row.extend(patient.QRS_interval["interval"][i])
    row.extend(patient.QRS_interval["paverage_ten"][i])
    row.extend(patient.QRS_interval["average_fifty"][i])

    row.extend(patient.P_Q_interval["interval"][i])
    row.extend(patient.P_Q_interval["paverage_ten"][i])
    row.extend(patient.P_Q_interval["average_fifty"][i])

    row.extend(patient.P_R_interval["interval"][i])
    row.extend(patient.P_R_interval["paverage_ten"][i])
    row.extend(patient.P_R_interval["average_fifty"][i])

    row.extend(patient.S_T_interval["interval"][i])
    row.extend(patient.S_T_interval["paverage_ten"][i])
    row.extend(patient.S_T_interval["average_fifty"][i])

    row.extend(patient.R_T_interval["interval"][i])
    row.extend(patient.R_T_interval["paverage_ten"][i])
    row.extend(patient.R_T_interval["average_fifty"][i])

    row.extend(patient.P_T_interval["interval"][i])
    row.extend(patient.P_T_interval["paverage_ten"][i])
    row.extend(patient.P_T_interval["average_fifty"][i])

    row.extend(patient.neg_P_Q_interval[i])
    row.extend(patient.neg_P_R_interval[i])
    row.extend(patient.neg_S_T_interval[i])
    row.extend(patient.neg_R_T_interval[i])
    row.extend(patient.neg_P_T_interval[i])
    row.extend(patient.P_neg_T_interval[i])
    row.extend(patient.neg_P_neg_T_interval[i])
    

    
    #time_lens.append(len(patient.segmented_beat_time[i]))
    #row.extend(mit100.segmented_beat_1[i])
    #beats_lens.append(len(patient.segmented_beat_1[i]))
        #if (len(patient.segmented_beat_1[i]) == 347):
            #print(patient.filename)
            #print(i)
            #mit207 = patient
                
    y[i] = patient.segmented_class_ID[i]
    x[i,0:columns] = row

    #print(DBn1[row_count])
    