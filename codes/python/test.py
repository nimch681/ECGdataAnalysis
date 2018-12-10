from scipy.signal import medfilt, lfilter, firwin, convolve
import numpy as np
import math
#from pymatbridge import Matlab as matlab

class ECG_FIR_filter:
    def __init__(self,medfilt_width_1=71,medfilt_width_2=215,is_low_pass=True, cutoff_fre=35, sampling_fre=360, fir_order=12):
        self.medfilt_width_1 = medfilt_width_1
        self.medfilt_width_2 = medfilt_width_2
        self.is_low_pass = is_low_pass
        self.cutoff_fre = cutoff_fre
        self.sampling_fre = sampling_fre
        self.fir_order = fir_order

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
  






#testing record 100 with denoising method  
mit100 = load_patient_record("mitdb","100")
filter = ECG_FIR_filter()
denoising_signal_FIR(mit100,filter)
display_signal_in_seconds(mit100,mit100.MLII,3)



def lgth_transform(ecg, ws):
    lgth=ecg.shape[0]
    return lgth

