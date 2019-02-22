#!/usr/bin/env python

"""
load_database.py written by Chontira Chumsaeng
TODO adapted from _______________
"""


import matplotlib.pyplot as plt
import os
from codes.python import heartbeat_segmentation as hs
from codes.python import ECG_denoising as denoise
import numpy as np
import csv
import math
import wfdb
from wfdb import processing

# Show a 2D plot with the data in beat
def display_signal(beat):
    plt.plot(beat)
    plt.ylabel('Signal')
    plt.show()
        

class Patient_record:
    def __init__(self,filename, database):
        self.database = database
        self.filename = filename
        self.fields = []
        self.time = []
        self.MLII = []
        self.filtered_MLII = []
        self.V1 = []
        self.filtered_V1 = []
        self.annotations = []
        self.annotated_R_poses = []
        self.annotated_beat_class = []
        self.annotated_p_waves_pos = []
        self.annotated_t_waves_pos = []
        self.segmented_class_ID = []
        self.segmented_beat_class = []
        self.segmented_R_pos = []
        self.original_R_pos = []
        self.segmented_beat_time = []
        self.segmented_beat_index = []
        self.segmented_beat_1 = []
        self.segmented_beat_2 = []
        self.Q_points = []
        self.S_points = []

    def attribute(self):
        print("database, filename, fields, time, MLII, filtered_MLII, V1, filtered_V1, annotations, annotated_R_poses, annotated_beat_class, annotated_p_waves_pos, annotated_t_waves_pos, segmented_class_ID, segmented_beat_class,segmented_R_pos, segmented_valid_R, segmented_original_R, segmented_beat_1, segmented_beat_2 ")
    
    def get_beat_1(self):
        return self.segmented_beat_1

    def get_beat_2(self):
        return self.segmented_beat_2
      
    def get_r_pos(self):
        return self.segmented_R_pos
    
    def set_segmented_beats_r_pos(self,filtered=True,is_MLII=True,is_V1=False,winL=180,winR=180,rr_max = 5):
        signal_MLII = []
        signal_V1 = []
        segmented_beat_class = []
        segmented_class_ID=[]
        segmented_R_pos = []
        beat_index = []
        times= []
        print("Start segmenting records: "+ self.filename)
        if(filtered == True):
            filter_FIR = denoise.ECG_FIR_filter()
            if(is_MLII == True):
                signal_MLII = denoise.denoising_signal_FIR(self.MLII,filter_FIR)
                self.filtered_MLII = signal_MLII
                print("Filtered MLII records from : "+ self.filename)

            if(is_V1 == True):
                signal_V1 =  denoise.denoising_signal_FIR(self.V1,filter_FIR)
                self.filtered_V1 = signal_V1
                print("Filtered V1 records from : "+ self.filename)
            
        else:
            signal_MLII = self.MLII
            signal_V1 = self.V1
        if(is_V1 == True):
            print("start segmenting V1.")
            segmented_beat_2, times,beat_index, segmented_beat_class, segmented_class_ID, segmented_R_pos,orignal_r_pos  = hs.segment_beat_from_annotation(signal_V1, self.time, self.annotations, winL, winR, rr_max)
            self.segmented_beat_2 = segmented_beat_2
            print("Finished segmenting V1.")
        if(is_MLII == True):
            print("start segmenting MLII.")
            segmented_beat_1,times,beat_index, segmented_beat_class, segmented_class_ID, segmented_R_pos,orignal_r_pos  = hs.segment_beat_from_annotation(signal_MLII, self.time, self.annotations, winL, winR, rr_max)
            self.segmented_beat_1 = segmented_beat_1
            print("Finished segmenting MLII.")
            
        self.segmented_beat_time = times
        self.segmented_beat_index = beat_index
        self.segmented_beat_class = segmented_beat_class
        self.segmented_class_ID = segmented_class_ID
        self.segmented_R_pos = segmented_R_pos
        self.original_R_pos = orignal_r_pos

        print("Segmenting record "+ self.filename + " completes.")
    
   # def set_segmented_s_and_q(self, R_peaks, time_limit = 0.01, limit=50):
        #if(self.filtered_MLII == []):
            #self


            
        
 
class ecg_database:
    def __init__(self,database):
        # Instance atributes v
        self.database = database
        self.patient_records = []
        self.filenames = []
        self.MITBIH_classes = []
        self.AAMI_classes = []
        self.data_for_classification = []
        #self.beat = np.empty([]) # record, beat, lead
        #self.class_ID = []   
        #self.valid_R = []       
        #self.R_pos = []
        #self.orig_R_pos = []
    
    def set_MIT_class(self,mitbih_classes):
        self.MITBIH_classe = mitbih_classes
    
    def set_AAMI_classes(self, classes):
        self.AAMI_classes = classes

    def attribute(self):
        print(" database, patient_records, MITBIH_classes, AAMI_classes ")
    
    def segment_beats(self,filtered=True,is_MLII=True,is_V1=False,winL=180,winR=180,rr_max = 5):

        
        for record in self.patient_records:
            record.set_segmented_beats_r_pos(filtered,is_MLII,is_V1,winL,winR,rr_max)
 
        print("Segmenting beats complete")

    



        

def create_ecg_database(database,patient_records):
    db = ecg_database(database)
    record_list = load_cases_from_list(database, patient_records)
    db.patient_records = record_list
    db.filenames = patient_records
    return db


def load_mitdb():


    mitdblstring = wfdb.get_record_list("mitdb")
    mitdbls = [int(i) for i in mitdblstring]
    mitdb = []


    for i in mitdbls:
        mitdb.append(load_patient_record("mitdb", str(i)))       
    my_db = ecg_database("mitdb")

    MITBIH_classes = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F']#, 'P', '/', 'f', 'u']
    AAMI_classes = [] 
    AAMI_classes.append(['N', 'L', 'R'])                    # N
    AAMI_classes.append(['A', 'a', 'J', 'S', 'e', 'j'])     # SVEB 
    AAMI_classes.append(['V', 'E'])                         # VEB
    AAMI_classes.append(['F'])                              # F

   

    my_db.patient_records = mitdb
    my_db.MITBIH_classes = MITBIH_classes
    my_db.AAMI_classes = AAMI_classes
    my_db.filenames = mitdblstring
    #my_db.Annotations = annotations  
    return my_db



def load_patient_record(DB_name, record_number):
    patient_record = Patient_record(record_number, DB_name)
    pathDB = os.getcwd()+'/database/'
    filename = pathDB + DB_name +"/"+ record_number
    print(filename)
    sig, fields = wfdb.rdsamp(filename, channels=[0,1])
    filename = pathDB + DB_name + "/csv/" + record_number +".csv"
    print(filename)
    f = open(filename, "r")
    reader = csv.reader(f, delimiter=',')
    next(reader) # skip first line!
    next(reader)
    time = []
    p_waves_pos = []
    t_waves_pos =[]
    MLII_index = 0
    V1_index = 1
    if int(record_number) == 114:
        MLII_index = 1
        V1_index = 0

    #MLII = []
    #V1 = []
    #time = []
    for row in reader:
        time.append((float(row[0])))
        #MLII.append((float(row[MLII_index])))
        #V1.append((float(row[V1_index])))
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

    filename = pathDB + DB_name + "/p_t_wave/" + record_number +"pt.csv"
    print(filename)
    f = open(filename, "r")
    reader = csv.reader(f, delimiter=',')
    for line in reader:
    
        if (float(line[0]) == -1):
            break    
        p_waves_pos.append(float(line[0]))

    for line in reader:
        t_waves_pos.append(float(line[0]))
    
    f.close

    patient_record.filename = record_number
    patient_record.fields = fields
    patient_record.time = time
    patient_record.annotated_p_waves_pos = p_waves_pos
    patient_record.annotated_t_waves_pos = t_waves_pos
    patient_record.MLII = sig[:,MLII_index] 
    patient_record.V1 = sig[:,V1_index] 
    patient_record.annotations = annotations
    patient_record.annotated_R_poses = annotated_orignal_R_poses
    patient_record.annotated_beat_class = annotated_beat_type
    

    return patient_record

def load_cases_from_list(database,patient_list):
    record_list = []
    for p in patient_list:
        patient = load_patient_record(database, str(p))
        record_list.append(patient)
    return record_list

def display_signal_in_seconds(patient_record,signal, time_in_second):
    sum = 0
    new_signal = []
    for t in range(0,len(signal)):
        #print(mit100.time[t+1])
        if(sum <= time_in_second):
            sum= patient_record.time[t] + patient_record.time[t+1]
            new_signal.append(signal[t])

    display_signal(new_signal)





    


    


