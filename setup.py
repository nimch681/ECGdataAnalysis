#from codes.python import simple_heartbeat_segmentation as shs
from codes.python import load_database,ECG_denoising
from codes.python import QRS_detector
import numpy as np
from scipy.signal import resample
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import neurokit as nk
from scipy.signal import savgol_filter
from biosppy.signals import ecg
#mitdb = load_mitdb.load_mitdb()




r_pos = np.array(DB1.patient_records[2].segmented_R_pos)

wfdb.plot_items(signal=DB1.patient_records[2].MLII, ann_samp=[r_pos])

time100 = np.array(mit100.segmented_beat_time)
beats100 = np.array(mit100.segmented_beat_1)
beat_index = np.array(mit100.segmented_beat_index)

wfdb.plot_items(signal=mit100.segmented_beat_1[0])

load_database.display_signal(mit100.segmented_beat_1[2])

mit1_10000 = mit100.filtered_MLII[0:1000]
wfdb.plot_items(signal=mit1_10000)


patient_list_1 = ["101","106","108","109","112","114","115","116","118","119","122","124","201","203","205","207","208","209","215","220","223","230"]
patient_list_2 = ["100","103","105","111","113","117","121","123","200","202","210","212","213","214","219","221","222","228","231","232","233","234"]
DB1 = load_database.create_ecg_database("mitdb",patient_list_1)
DB2 = load_database.create_ecg_database("mitdb",patient_list_2)
DB1.segment_beats()
DB2.segment_beats()
DB1.set_Q_and_S_points()
DB2.set_Q_and_S_points()

mit105 = load_database.load_patient_record("mitdb","105")
mit105.set_segmented_beats_r_pos(winL=100,winR=200)
r_pos = np.asarray(mit105.segmented_R_pos)
wfdb.plot_items(signal=mit105.filtered_MLII,ann_samp=[r_pos])



r_pos = np.asarray(mit100.original_R_pos)

wfdb.plot_items(signal=mit100.filtered_MLII,ann_samp=[r_pos])



mit100 = load_database.load_patient_record("mitdb","100")
mit100.set_segmented_beats_r_pos(winL=100,winR=200)
out = ecg.ecg(signal=mit100.filtered_MLII[0:20000], sampling_rate=1000., show=True)

wfdb.plot_items(signal=mit100.filtered_MLII)
wfdb.plot_items(signal=mit100.MLII[0:400])
wfdb.plot_items(signal=mit100.filtered_MLII[0:300])
filtered_np100 = savgol_filter(mit100.filtered_MLII[0:1000],51,7)
wfdb.plot_items(signal=filtered_np100)

mit100.set_Q_S_points_MLII()
columns = len(mit100.segmented_beat_time[0]) + len(mit100.segmented_beat_1[0])
rows = 0
for patient in DB1.patient_records:
        rows += len(patient.segmented_beat_time)

DBn1 = np.zeros((rows,columns),dtype=object)
yn1 = np.zeros((rows,1), dtype=object)
q_points = np.zeros((rows,1), dtype=int)
s_points = np.zeros((rows,1), dtype=int)

row_count = 0

for patient in DB1.patient_records:
        
        for i in range(0,len(patient.segmented_beat_time)):
               
                row_x = list()
                row_x.extend(patient.segmented_beat_time[i])
                #time_lens.append(len(patient.segmented_beat_time[i]))
                row_x.extend(patient.segmented_beat_1[i])
                #beats_lens.append(len(patient.segmented_beat_1[i]))
                #if (len(patient.segmented_beat_1[i]) == 347):
                        #print(patient.filename)
                        #print(i)
                        #mit207 = patient
                

                yn1[row_count] = patient.segmented_class_ID[i]
                DBn1[row_count,0:columns] = row_x  
                #print(DBn1[row_count])
                row_count += 1
                
                #class_lens.append(len(patient.segmented_beat_class[i]))
                #if(i == 0):
                        #DBn1 = np.array(row, dtype=object)
                #else:
                        #DBn1 = np.vstack((DBn1, row))

"""
wfdb.plot_items(signal=mit207.MLII)
load_database.display_signal(beat=mit207.segmented_beat_1[2382])


mit207 = load_database.load_patient_record("mitdb","207")
mit207.set_segmented_beats_r_pos(filtered=False)
len(mit207.segmented_beat_1[2381])
len(mit207.segmented_beat_1[2381])
len(mit207.segmented_beat_1[2383])
mit207.segmented_R_pos[2382]
len(mit207.MLII)
len(mit207.segmented_beat_1)
false_beat = shs.segment(mit207.MLII,mit207.segmented_R_pos[2382],180,180,35)

649799- 649652

DBn1 = np.array(DB1list)

for i in range(0, len(beats_lens)-1):
        if(beats_lens[i] != beats_lens[i+1] ):
                print("this index is different" + str(i))

beats_lens[35831]

for i in range(0, len(time_lens)-1):
        if(time_lens[i] != time_lens[i+1] ):
                print("this index is different" + str(i))

"""

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


pca = PCA(n_components=2)

projected = pca.fit_transform(DBn1)
x_test = pca.fit_transform(DBn2)

yn1=yn1.astype('int')
yn2 = yn2.astype('int')
print(pca.explained_variance_ratio_)
targets = pd.DataFrame(data = yn1, columns = ['target'])
principalDf = pd.DataFrame(data = projected, columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf,targets], axis = 1)

clf = svm.SVC()

clf.fit(projected, yn1.ravel()) 

clf.score(x_test,yn2)





#row = np.array(row, dtype=object)
#row = np.array(row)
        #print(row.shape)
        
       # np.stack((DB1np, row), axis=0)
        #np.stack((DB1np, patient.segmented_beat_1), axis=-1)
        #np.stack((DB1np, patient.segmented_beat_2), axis=-1)






#segmenting record 100
"""  
load_mitdb.display_signal_in_seconds(mit100,mit100.MLII,3)
filter_ecg = ECG_denoising.ECG_FIR_filter()
mit100.filtered_MLII = ECG_denoising.denoising_signal_FIR(mit100.MLII,filter_ecg)
mit100.filtered_V1 = ECG_denoising.denoising_signal_FIR(mit100.V1,filter_ecg)
mit100.segmented_beat_1, mit100.segmented_class_ID, mit100.segmented_beat_class, mit100.segmented_R_pos, mit100.segmented_valid_R, mit100.segmented_original_R  = shs.segment_beat(mit100.filtered_MLII, mit100.time, mit100.annotations, 90, 90)
#np100R = np.concatenate((mit100.segmented_original_R,mit100.segmented_R_pos,mit100.segmented_valid_R ))

MLII = np.array(mit100.MLII)
np100_MLII = np.array(mit100.segmented_beat_1)

np100R = np.array(mit100.segmented_original_R)
np_r_poses = np.array(mit100.segmented_R_pos)
np_valid_r = np.array(mit100.segmented_valid_R)
#np100R = np.append(np100R, np_r_poses, axis=1)
np100R = np.column_stack((np100R, np_r_poses))
np100R = np.column_stack((np100R, np_valid_r))
"""

#checking highest peaks in record 100


"""
count = 0
for i in range(0,len(np100R)):
    ori_R =np100R[i,1]
    seg_R = np100R[i,3]
    if(ori_R <= seg_R):
        count = count+ 1

percent = (len(np100R)/count) * 100
"""
#resampling code
"""
Resampled_MLII = []

def resample_signal(fs):
        factor = 360.0 / fs
        num_samples = int(round(factor * len(mit100.MLII)))
        Resampled_MLII = resample(mit100.MLII, num_samples)

factor = 360.0 / 200
num_samples = int(round(factor * len(mit100.MLII)))
Resampled_MLII = resample(mit100.MLII, num_samples)

index, value = max(enumerate(MLII), key=operator.itemgetter(1))

"""









