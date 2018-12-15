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
import os
import csv
import math

# Show a 2D plot with the data in beat
def display_signal(beat):
    plt.plot(beat)
    plt.ylabel('Signal')
    plt.show()
        

class Patient_record:
    def __init__(self,filename, database):
        self.database = database
        self.filename = filename
        self.time = []
        self.MLII = []
        self.filtered_MLII = []
        self.V1 = []
        self.filtered_V1 = []
        self.annotations = []
        self.annotated_R_poses = []
        self.annotated_beat_class = []
        self.segmented_class_ID = []
        self.segmented_beat_class = []
        self.segmented_R_pos = []
        self.segmented_valid_R = []
        self.segmented_original_R = []
        self.segmented_beat_1 = []
        self.segmented_beat_2 = []

    def attribute(self):
        print("database, filename, time, MLII, filtered_MLII, V1, filtered_V1, annotations, annotated_R_poses, annotated_beat_class, segmented_class_ID, segmented_beat_class,segmented_R_pos, segmented_valid_R, segmented_original_R, segmented_beat_1, segmented_beat_2 ")
        

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

    def attribute(self):
        print("filename, database, Annotations, MITBIH_classes, AAMI_classes ")
        


def load_mitdb():
    my_db = ecg_database("mitdb")
    pathDB = os.getcwd()+'/database/'
    DB_name = 'mitdb'
    fs = 360
    jump_lines = 1

    #Read files: signal (.csv )  annotations (.txt)    
    fRecords = list()
    fAnnotations = list()

    lst = os.listdir(pathDB + DB_name + "/csv")
    lst.sort()
    fRecords = list()
    fAnnotations = list()
    for file in lst:
        if file.endswith(".csv"):
       
            fRecords.append(file)
        elif file.endswith(".txt"):
      
            fAnnotations.append(file)        

    MITBIH_classes = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F']#, 'P', '/', 'f', 'u']
    AAMI_classes = []
    AAMI_classes.append(['N', 'L', 'R'])                    # N
    AAMI_classes.append(['A', 'a', 'J', 'S', 'e', 'j'])     # SVEB 
    AAMI_classes.append(['V', 'E'])                         # VEB
    AAMI_classes.append(['F'])                              # F

    RAW_signals = []
    r_index = 0

    my_db.filename = fRecords
    my_db.Annotations = fAnnotations
    my_db.MITBIH_classes = MITBIH_classes
    my_db.AAMI_classes = AAMI_classes
    #my_db.Annotations = annotations  
    return my_db



def load_patient_record(DB_name, record_number):
    patient_record = Patient_record(record_number, DB_name)
    pathDB = os.getcwd()+'/database/'
    filename = pathDB + DB_name + "/csv/" + record_number +".csv"
    print(filename)
    f = open(filename, "r")
    reader = csv.reader(f, delimiter=',')
    next(reader) # skip first line!
    next(reader)
    MLII_index = 1
    V1_index = 2
    if int(record_number) == 114:
        MLII_index = 2
        V1_index = 1

    MLII = []
    V1 = []
    time = []
    for row in reader:
        time.append((float(row[0])))
        MLII.append((float(row[MLII_index])))
        V1.append((float(row[V1_index])))
    f.close

    filename = pathDB + DB_name + "/csv/" + record_number +".txt"
    print(filename)
    f = open(filename, 'rt')
    next(f) # skip first line!

    annotations = []
    for line in f:
        annotations.append(line)
    f.close

    annotated_beat_type = []
    annotated_orignal_R_poses = []

    for a in annotations:
    
        aS = a.split()
            
        annotated_orignal_R_poses.append(int(aS[1]))
        annotated_beat_type.append(str(aS[2]))

    patient_record.filename = record_number
    patient_record.time = time
    patient_record.MLII = MLII
    patient_record.V1 = V1
    patient_record.annotations = annotations
    patient_record.annotated_R_poses = annotated_orignal_R_poses
    patient_record.annotated_beat_class = annotated_beat_type

    return patient_record



def display_signal_in_seconds(patient_record,signal, time_in_second):
    sum = 0
    new_signal = []
    for t in range(0,len(signal)):
        #print(mit100.time[t+1])
        if(sum <= time_in_second):
            sum= patient_record.time[t] + patient_record.time[t+1]
            new_signal.append(signal[t])

    display_signal(new_signal)


if __name__ == "__main__":
    my_mitdb = load_mitdb()
    


    


