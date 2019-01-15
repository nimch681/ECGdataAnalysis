from codes.python import simple_heartbeat_segmentation as shs
from codes.python import load_mitdb,ECG_denoising
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

mitdblstring = wfdb.get_record_list("mitdb")
mitdbls = [int(i) for i in mitdblstring]
mitdb = []

for i in mitdbls:
    mitdb.append(load_mitdb.load_patient_record("mitdb", str(i)))
    
mit100 = mitdb[0]
#mitdb = load_mitdb.load_mitdb()
#mit100 = load_mitdb.load_patient_record("mitdb","100")
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


count = 0
for i in range(0,len(np100R)):
    ori_R =np100R[i,1]
    seg_R = np100R[i,3]
    if(ori_R <= seg_R):
        count = count+ 1

percent = (len(np100R)/count) * 100

Resampled_MLII = []
    
def resample_signal(fs):
        factor = 360.0 / fs
        num_samples = int(round(factor * len(mit100.MLII)))
        Resampled_MLII = resample(mit100.MLII, num_samples)

factor = 360.0 / 200
num_samples = int(round(factor * len(mit100.MLII)))
Resampled_MLII = resample(mit100.MLII, num_samples)

index, value = max(enumerate(MLII), key=operator.itemgetter(1))

np100R.shape
np100R



#load_mitdb.display_signal_in_seconds(mit100,mit100.filtered_MLII,3)








