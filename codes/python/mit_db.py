#!/usr/bin/env python

"""
mit_db.py
Description:
Contains the classes for store the MITBIH database and some utils
VARPA, University of Coruna
Mondejar Guerra, Victor M.
24 Oct 2017
"""

import matplotlib.pyplot as plt
import numpy as np

# Show a 2D plot with the data in beat
def display_signal(beat):
    plt.plot(beat)
    plt.ylabel('Signal')
    plt.show()

# Class for RR intervals features
class RR_intervals:
    def __init__(self):
        # Instance atributes
        self.pre_R = np.array([])
        self.post_R = np.array([])
        self.local_R = np.array([])
        self.global_R = np.array([])

        

class Patient_record:
    def __init__(self,filename, database):
        self.database = database
        self.filename = filename
        self.time = []
        self.MLII = []
        self.denoisedMLII = []
        self.V1 = []
        self.denoisedV1 = []
        self.annotations = []

class ECG_FIR_filter:
    def __init__(self,medfilt_width_1, medfilt_width_2,is_low_pass, low_pass_fre):
        self.medfilt_width_1 = medfilt_width_1
        self.medfilt_width_2 = medfilt_width_2
        self.is_low_pass = is_low_pass
        self.low_pass_fre = low_pass_fre

class ecg_database:
    def __init__(self, database):
        # Instance atributes
        self.filename = []
        self.database = database
        self.Annotations =[]
        self.MITBIH_classes = []
        self.AAMI_classes = []
        #self.beat = np.empty([]) # record, beat, lead
        #self.class_ID = []   
        #self.valid_R = []       
        #self.R_pos = []
        #self.orig_R_pos = []