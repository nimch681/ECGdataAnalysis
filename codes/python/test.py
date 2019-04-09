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



mit100 = DB2.patient_records[0]
t_points = np.asarray(mit100.T_points[0:5],dtype=int)
p_points = np.asarray(mit100.P_points[0:5],dtype=int)

wfdb.plot_items(signal=mit100.filtered_MLII[0:2000],ann_samp=[p_points])

mit100.set_r_properties_MLII()
mit100.set_Q_S_points_MLII()
mit100.set_P_T_points_MLII()

rr_prop = rr_interval_and_average(mit100)



QRS_properties, P_Q_properties, P_Q_neg, P_R_properties, P_R_neg, S_T_properties, S_T_neg,R_T_properties, R_T_neg, P_T_properties,neg_P_T, P_T_neg, neg_P_T_neg =interval_and_average(mit100)

