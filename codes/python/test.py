import os
import csv
from scipy.signal import medfilt, lfilter, firwin, convolve
from pymatbridge import Matlab as matlab
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
        elif file.endswith(".text"):
      
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

    filename = pathDB + DB_name + "/csv/" + record_number +".text"
    print(filename)
    f = open(filename, 'rb')
    next(f) # skip first line!

    annotations = []
    for line in f:
        annotations.append(line)
    f.close
    patient_record.filename = record_number
    patient_record.time = time
    patient_record.MLII = MLII
    patient_record.V1 = V1
    patient_record.annotations = annotations

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




def denoising_signal_FIR(patient_record, FIR_filter):
    baseline = medfilt(patient_record.MLII, FIR_filter.medfilt_width_1) #has to be an odd number (360hz*0.2second)
    baseline = medfilt(baseline, FIR_filter.medfilt_width_2) #has to be an odd number (360hz*0.6second)

    denoisedMLII = []
    denoisedV1 = []
    # Remove Baseline
    for i in range(0, len(patient_record.MLII)):
        denoisedMLII.append(patient_record.MLII[i] - baseline[i])


            # median_filter1D
    baseline = medfilt(patient_record.V1, FIR_filter.medfilt_width_1) 
    baseline = medfilt(baseline, FIR_filter.medfilt_width_2) 


            # Remove Baseline
    for i in range(0, len(patient_record.V1)):
        denoisedV1.append(patient_record.V1[i] - baseline[i])

    if(FIR_filter.is_low_pass == True):
        FC = FIR_filter.cutoff_fre/(0.5*FIR_filter.sampling_fre)
        b = firwin(FIR_filter.fir_order, cutoff = FC, window = "hamming")
        denoisedMLII = convolve(denoisedMLII, b, mode='same')
        denoisedV1 = convolve(denoisedV1, b, mode='same')
    
    patient_record.filtered_MLII = denoisedMLII
    patient_record.filtered_V1 = denoisedV1
  


def pan_tompskin_QRS_detector():
    return

   # filename = pathDB + DB_name + "/csv/" + fAnnotations[record_number]
   # print(filename)
    #f = open(filename, 'rb')
    #next(f) # skip first line!

    #annotations = []
    #for line in f:
        #annotations.append(line)
   # f.close

#mit100 = load_patient_record("mitdb","100")
#baseline = medfilt(mit100.MLII, 71) #has to be an odd number (360hz*0.2second)
#baseline = medfilt(baseline, 215) #has to be an odd number (360hz*0.6second)



#testing record 100 with denoising method  
mit100 = load_patient_record("mitdb","100")
filter = ECG_FIR_filter()
denoising_signal_FIR(mit100,filter)
display_signal_in_seconds(mit100,mit100.MLII,3)




