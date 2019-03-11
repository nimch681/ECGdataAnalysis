
import numpy as np
from scipy import signal
from scipy.signal import savgol_filter
import operator
from numpy import array
from wfdb import processing, plot
from sklearn import metrics
import wfdb


def average(numbers):
    return float(sum(numbers)) / len(numbers)        

def peak_properties_extractor(sig,start_point=None,end_point=None,height=None, distance=None, width = None, plateau_size=None):
    sig = sig[start_point:end_point]
    peaks,properties  = np.asarray(signal.find_peaks(sig, height=height, distance=distance,width=width,plateau_size=plateau_size))
    return peaks,properties

def point_transform_to_origin(por,point):
    point_from_origin = por + point 
    return point_from_origin

def origin_to_new_point(por,point_from_origin):
    point = point_from_origin - por
    return point

def peak_duration(time,right_edge, left_edge,point_from_origin=0):
    right_edge = point_transform_to_origin(point_from_origin,right_edge)
    left_edge = point_transform_to_origin(point_from_origin,left_edge)
    
    return float(time[right_edge]-time[left_edge])

def sub_signal_interval(time, start_point, end_point,point_from_origin=0):
    start_point = point_transform_to_origin(point_from_origin,start_point)
    end_point = point_transform_to_origin(point_from_origin,end_point)
    
    return float(time[end_point]-time[start_point])

def peak_height(signal, peak, prominence,point_from_origin=0):
    peak = point_transform_to_origin(point_from_origin,peak)
    height = signal[peak]-(signal[peak] - prominence)
    return height

def area_under_curve(signal,time,samples,point_from_origin=0):
    samples = [point_transform_to_origin(i,point_from_origin) for i in samples]
    time = np.asarray(time)
    amplitude = np.asarray(signal)
    area = metrics.auc(time[samples],amplitude[samples])
    return area

def amplitude(signal,samples,point_from_origin=0):
    samples = [point_transform_to_origin(i,point_from_origin) for i in samples]
    signal = np.asarray(signal)
    amplitudes = signal[samples]
    return amplitudes
    
def pre_pos_rr_interval(record):
    pre_r = record.segmented_R_pos[0]
    current_r = 0
    post_r = 0 
    pre_rr_interval = []
    post_rr_interval = []
    
    for r in range(0, len(record.segmented_R_pos)):
        current_r = record.segmented_R_pos[r]
        
        if(r < len(record.segmented_R_pos)-1):
            post_r = record.segmented_R_pos[r+1]
        else:
            post_r = current_r
        
        pre_rr_interval.append(record.time[current_r] - record.time[pre_r])
        post_rr_interval.append(record.time[post_r] - record.time[current_r] )
        pre_r = current_r
        
        
    return  pre_rr_interval, post_rr_interval


def rr_average_by_sample(pre_rr_interval,ten=True,fifty=False, all=True):
     
    rr_ten = []
    rr_fifty = []
    rr_all = []
    num_max_beat = len(pre_rr_interval)
    if all == True:
        rr_average_all = rr_global_average(pre_rr_interval)
        rr_all = [rr_average_all] * num_max_beat

    for i in range(0,num_max_beat):
        rr_average_10 = 0
        rr_average_50 = 0
        if ten == True:
            rr_average_10=rr_local_average(pre_rr_interval,num_max_beat, i, 10)
            
            rr_ten.append(rr_average_10)
                
        if fifty == True:
            
            rr_average_10=rr_local_average(pre_rr_interval,num_max_beat, i, 50)
            
            rr_fifty.append(rr_average_50)

        
    return rr_ten, rr_fifty, rr_all

def rr_global_average(pre_rr_interval):
    return average(pre_rr_interval)

def rr_local_average(pre_rr_interval,len_of_rr, pos, average_of_rr):
    half_ave = int(average_of_rr/2)
    rr_ave = 0
    if pos >= len_of_rr-(half_ave+1):
        rr_ave = average(pre_rr_interval[pos-half_ave:len_of_rr-1])
    if pos <= half_ave:
        rr_ave = average(pre_rr_interval[1:pos+(half_ave+1)])
    if (pos > half_ave and pos < len_of_rr-(half_ave+1)):
        rr_ave = average(pre_rr_interval[pos-half_ave:pos+(half_ave+1)])
    
    return rr_ave


def r_peak_properties_extractor(patient,sample_from_R=[0,11], to_area=True, left_limit=50,right_limit=50, distance=20, width=[0,100],plateau_size=[0,100]):
    
    peaks = []
    heights = []
    durations = []
    areas = []
    onset = []
    offset = []
    amps = []
    promi = []
    time = patient.time
    print("Patient file: ",patient.filename, "begins")
    
    if(patient.filtered_MLII == []):
        print("Please filter the signal")
        return
    if(patient.segmented_R_pos == []):
        print("please segment the signal to find R peak")
        return
    
    for r in patient.segmented_R_pos:
        start_point = r-left_limit
        end_point = r+right_limit
        MLII = []
        sig = []
        if(patient.filtered_MLII[r] >= 0):
            MLII = patient.filtered_MLII
            sig = MLII[start_point:end_point]
            height = min(sig)   
            peak,properties = peak_properties_extractor(sig,height=height, distance=distance, width = width, plateau_size=plateau_size)
        else:
            MLII = -patient.filtered_MLII
            sig = MLII[start_point:end_point]
            height = min(sig)   
            peak,properties = peak_properties_extractor(sig,height=height, distance=distance, width = width, plateau_size=plateau_size)
    
    
        savgol_signal = savgol_filter(sig,41,9)
        height = min(savgol_signal)   
        peak_savol,properties_savol = peak_properties_extractor(savgol_signal,height=height, distance=distance, width=width, plateau_size=plateau_size)
        old_sig = savgol_signal
        savgol_signal = savgol_signal[peak_savol]
       
        value = max(savgol_signal)
        index = np.where(savgol_signal==value)
        
        index = int(index[0])
    
        peak_savol = peak_savol[index]
        peak_savol = point_transform_to_origin(start_point,peak_savol)
        
        left_ips = np.asarray(properties_savol["left_ips"])
        right_ips = np.asarray(properties_savol["right_ips"])
        left_ips = [int(i) for i in left_ips]
        right_ips = [int(i) for i in right_ips]
    
        left_edge = left_ips[index]
        right_edge = right_ips[index]
        duration = round(peak_duration(time=patient.time,right_edge=right_edge, left_edge=left_edge,point_from_origin=start_point),3)
        prominences = np.asarray(properties_savol["prominences"])
        prominence = prominences[index]
        height = round(peak_height(MLII, peak_savol, prominence,0),3)
        peaks.append(peak_savol)
        durations.append(duration)
        promi.append(prominence)
        amp = amplitude(patient.filtered_MLII,list(range(sample_from_R[0],sample_from_R[1])),start_point)
        heights.append(height)
        if(to_area==True):
            samples = list(range(left_edge,right_edge+1))
            area = round(area_under_curve(patient.filtered_MLII,time,samples,start_point),3)
            areas.append(area)
        amps.append(amp)
        offset.append(right_edge+5)
        onset.append(left_edge+5)
    properties = {
        "peaks" : peaks,
        "durations" : durations,
        "prominences" : promi,
        "height" : heights,
        "amplitudes" : amps,
        "areas" : areas,
        "onset" : onset,
        "offset" : offset
        }
    print("Patient file: ",patient.filename, "processing end")
    return properties    
    










