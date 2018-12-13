import numpy as np 
import math
import matplotlib.pyplot as plt
from numpy import array
from codes.python import QRS_detector
import sys
import csv
import os
import operator
import matplotlib.pyplot as plt
import wfdb
mit100 = load_mitdb.load_patient_record("mitdb","100")
filter_ecg = ECG_denoising.ECG_FIR_filter()
mit100.filtered_MLII = ECG_denoising.denoising_signal_FIR(mit100.MLII,filter_ecg)
mit100.filtered_V1 = ECG_denoising.denoising_signal_FIR(mit100.V1,filter_ecg)
#mit100.segmented_beat_1, mit100.segmented_class_ID, mit100.segmented_beat_class, mit100.segmented_R_pos = shs.segment_beat(mit100.filtered_MLII, mit100.time, mit100.annotations, 90, 90)
beats = mit100.beat_1

beat_limits = []

for i in range(0, len(beats)):
    limit_1 = beats[i][0][0]
    limit_2 = beats[i][0][len(beats[i][0])-1]
    beat_limits.append((limit_1,limit_2))

beat_time_limits = []

for i in range(0, len(beats)):
    limit_1 = beats[i][1][0]
    limit_2 = beats[i][1][len(beats[i][1])-1]
    beat_time_limits.append((limit_1,limit_2))
    

#mit100.beat_1[0][0:len(mit100.beat_1[0])]
#numbers = list(range(1, 1000))
#numbers
#len(mit100.beat_1[0])




