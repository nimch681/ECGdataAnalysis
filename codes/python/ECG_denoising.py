
from scipy.signal import medfilt, lfilter, firwin, convolve
import numpy as np
import math
from scipy.signal import savgol_filter

#from pymatbridge import Matlab as matlab

class ECG_FIR_filter:
    def __init__(self,medfilt_width_1=71,medfilt_width_2=215,is_low_pass=True, cutoff_fre=35, sampling_fre=360, fir_order=12):
        self.medfilt_width_1 = medfilt_width_1
        self.medfilt_width_2 = medfilt_width_2
        self.is_low_pass = is_low_pass
        self.cutoff_fre = cutoff_fre
        self.sampling_fre = sampling_fre
        self.fir_order = fir_order

    def attribute(self):
        print("medfilt_width_1, medfilt_width_2, is_low_pass, cutoff_fre, sampling_fre, fir_order")
      

def denoising_signal_FIR(signal, FIR_filter):
    baseline = medfilt(signal, FIR_filter.medfilt_width_1) #has to be an odd number (360hz*0.2second)
    baseline = medfilt(baseline, FIR_filter.medfilt_width_2) #has to be an odd number (360hz*0.6second)

    denoised_signal = []
    
    # Remove Baseline
    for i in range(0, len(signal)):
        denoised_signal.append(signal[i] - baseline[i])

    if(FIR_filter.is_low_pass == True):
        FC = FIR_filter.cutoff_fre/(0.5*FIR_filter.sampling_fre)
        b = firwin(FIR_filter.fir_order, cutoff = FC, window = "hamming")
        denoised_signal = convolve(denoised_signal, b, mode='same')
        
    
    return denoised_signal
  

class savgol:
    def __init__(self,window_len=17,poly_order=4):
        self.window_len = window_len
        self.poly_order = poly_order
      

    def attribute(self):
        print("window_len, poly_order")

def savol_filter(signal, savgol):
   
    
    denoised_signal = savgol_filter(signal,savgol.window_len,savgol.poly_order)
    
    return denoised_signal

if __name__ == "__main__":
    filter = ECG_FIR_filter()



