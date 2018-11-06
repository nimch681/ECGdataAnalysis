import numpy as np
import scipy as sp

from os import listdir, mkdir, system
from os.path import isfile, isdir, join, exists
import wfdb
import pandas as pd
import os


headers = ['time','MlII','V5']
mit100 = pd.read_csv("C:\\Users\\nimch681\\Documents\\git_projects\\ECGdataAnalysis\\database\\mitdb\\csv\\100.csv", names=headers)


mit100['time'] = mit100['time'].map(lambda x: datetime.strptime(str(x), '%Y/%m/%d %H:%M:%S.%f'))
x = df['time',]
y = df['MlII']



def load_signal(DS, winL, winR, do_preprocess):

    class_ID = [[] for i in range(len(DS))]
    beat = [[] for i in range(len(DS))] # record, beat, lead
    R_poses = [ np.array([]) for i in range(len(DS))]
    Original_R_poses = [ np.array([]) for i in range(len(DS))]   
    valid_R = [ np.array([]) for i in range(len(DS))]
    my_db = mit_db()
    patients = []

    # Lists 
    # beats = []
    # classes = []
    # valid_R = np.empty([])
    # R_poses = np.empty([])
    # Original_R_poses = np.empty([])

    size_RR_max = 20

    pathDB = '/home/mondejar/dataset/ECG/'
    DB_name = 'mitdb'
    fs = 360
    jump_lines = 1

    # Read files: signal (.csv )  annotations (.txt)    
    fRecords = list()
    fAnnotations = list()

    lst = os.listdir(pathDB + DB_name + "/csv")
    lst.sort()
    for file in lst:
        if file.endswith(".csv"):
            if int(file[0:3]) in DS:
                fRecords.append(file)
        elif file.endswith(".txt"):
            if int(file[0:3]) in DS:
                fAnnotations.append(file)        

    MITBIH_classes = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F']#, 'P', '/', 'f', 'u']
    AAMI_classes = []
    AAMI_classes.append(['N', 'L', 'R'])                    # N
    AAMI_classes.append(['A', 'a', 'J', 'S', 'e', 'j'])     # SVEB 
    AAMI_classes.append(['V', 'E'])                         # VEB
    AAMI_classes.append(['F'])                              # F
    #AAMI_classes.append(['P', '/', 'f', 'u'])              # Q

    RAW_signals = []
    r_index = 0

    #for r, a in zip(fRecords, fAnnotations):
    for r in range(0, len(fRecords)):

        print("Processing signal " + str(r) + " / " + str(len(fRecords)) + "...")

        # 1. Read signalR_poses
        filename = pathDB + DB_name + "/csv/" + fRecords[r]
        print (filename)
        f = open(filename, 'rb')
        reader = csv.reader(f, delimiter=',')
        next(reader) # skip first line!
        MLII_index = 1
        V1_index = 2
        if int(fRecords[r][0:3]) == 114:
            MLII_index = 2
            V1_index = 1

        MLII = []
        V1 = []
        for row in reader:
            MLII.append((int(row[MLII_index])))
            V1.append((int(row[V1_index])))
        f.close()


        RAW_signals.append((MLII, V1)) ## NOTE a copy must be created in order to preserve the original signal
        # display_signal(MLII)

        # 2. Read annotations
        filename = pathDB + DB_name + "/csv/" + fAnnotations[r]
        print(filename)
        f = open(filename, 'rb')
        next(f) # skip first line!

        annotations = []
        for line in f:
            annotations.append(line)
        f.close
        # 3. Preprocessing signal!
        if do_preprocess:
            #scipy.signal
            # median_filter1D
            baseline = medfilt(MLII, 71) 
            baseline = medfilt(baseline, 215) 

            # Remove Baseline
            for i in range(0, len(MLII)):
                MLII[i] = MLII[i] - baseline[i]

            # TODO Remove High Freqs

            # median_filter1D
            baseline = medfilt(V1, 71) 
            baseline = medfilt(baseline, 215) 

            # Remove Baseline
            for i in range(0, len(V1)):
                V1[i] = V1[i] - baseline[i]


        # Extract the R-peaks from annotations
        for a in annotations:
            aS = a.split()
            
            pos = int(aS[1])
            originalPos = int(aS[1])
            classAnttd = aS[2]
            if pos > size_RR_max and pos < (len(MLII) - size_RR_max):
                index, value = max(enumerate(MLII[pos - size_RR_max : pos + size_RR_max]), key=operator.itemgetter(1))
                pos = (pos - size_RR_max) + index

            peak_type = 0
            #pos = pos-1
            
            if classAnttd in MITBIH_classes:
                if(pos > winL and pos < (len(MLII) - winR)):
                    beat[r].append( (MLII[pos - winL : pos + winR], V1[pos - winL : pos + winR]))
                    for i in range(0,len(AAMI_classes)):
                        if classAnttd in AAMI_classes[i]:
                            class_AAMI = i
                            break #exit loop
                    #convert class
                    class_ID[r].append(class_AAMI)

                    valid_R[r] = np.append(valid_R[r], 1)
                else:
                    valid_R[r] = np.append(valid_R[r], 0)
            else:
                valid_R[r] = np.append(valid_R[r], 0)
            
            R_poses[r] = np.append(R_poses[r], pos)
            Original_R_poses[r] = np.append(Original_R_poses[r], originalPos)
        
        #R_poses[r] = R_poses[r][(valid_R[r] == 1)]
        #Original_R_poses[r] = Original_R_poses[r][(valid_R[r] == 1)]

        
    # Set the data into a bigger struct that keep all the records!
    my_db.filename = fRecords

    my_db.raw_signal = RAW_signals
    my_db.beat = beat # record, beat, lead
    my_db.class_ID = class_ID
    my_db.valid_R = valid_R
    my_db.R_pos = R_poses
    my_db.orig_R_pos = Original_R_poses

    return my_db
