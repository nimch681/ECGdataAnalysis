
import numpy as np
from scipy import signal
from scipy.signal import savgol_filter
import operator
from numpy import array
from wfdb import processing, plot
from sklearn import metrics
import wfdb
from math import *


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
    


## make change so that it will benefit the other signal as well

def r_peak_properties_extractor(patient,sample_from_R=[5,5], to_area=True,to_savol=True, Order=9,window_len=41, left_limit=50,right_limit=50, distance=1, width=[0,100],plateau_size=[0,100]):
    peaks = []
    heights = []
    durations = []
    areas = []
    onset = []
    offset = []
    amps = []
    promi = []
    sigs = []
    start_points = []
    end_points = []
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
        
        sigs.append(sig)
        if(to_savol == True):
            savgol_signal = savgol_filter(sig,window_len,Order)
        else:
            savgol_signal = sig
        
        
        height = min(savgol_signal)   
        peak_savol,properties_savol = peak_properties_extractor(savgol_signal,height=height, distance=distance, width=width, plateau_size=plateau_size)
        old_sig = savgol_signal
        old_peaks = peak_savol
        new_peaks = point_transform_to_origin(start_point,peak_savol)
        
        peak_savol = []

        for p in range(0,len(old_peaks)):
            if(new_peaks[p] >= r-5 and new_peaks[p] <= r+5):
                peak_savol.append(old_peaks[p])
                
        savgol_signal = savgol_signal[peak_savol]
        
        if(len(savgol_signal) == 0):
            start_points.append(start_point)
            end_points.append(end_point)
            peaks.append(r)
            durations.append(0)
            promi.append(0)
            amp = amplitude(patient.filtered_MLII,list(range(r-sample_from_R[0],r+sample_from_R[1])),0)
            heights.append(0)
            if(to_area==True):
                areas.append(0)
            amps.append(amp)
            offset.append(r+5)
            onset.append(r-5)
            
            
            
            continue
            
        
        value = max(savgol_signal)
            
        index = np.where(savgol_signal==value)
        
        index = int(index[0])
    
        peak_savol = peak_savol[index]
        r_peak = peak_savol
        peak_savol = point_transform_to_origin(start_point,peak_savol)
        
        left_ips = np.asarray(properties_savol["left_ips"])
        right_ips = np.asarray(properties_savol["right_ips"])
        left_ips = [int(i) for i in left_ips]
        right_ips = [int(i) for i in right_ips]

        index = np.where(old_peaks==r_peak)
        index = int(index[0])
    
        left_edge = left_ips[index]
        right_edge = right_ips[index]
        duration = round(peak_duration(time=patient.time,right_edge=right_edge, left_edge=left_edge,point_from_origin=start_point),3)
        prominences = np.asarray(properties_savol["prominences"])
        prominence = prominences[index]
        height = round(peak_height(MLII, peak_savol, prominence,0),3)
        peaks.append(peak_savol)
        durations.append(duration)
        promi.append(prominence)
        amp = amplitude(patient.filtered_MLII,list(range(r-sample_from_R[0],r+sample_from_R[1])),0)
        heights.append(height)
        if(to_area==True):
            samples = list(range(left_edge,right_edge+1))
            area = round(area_under_curve(patient.filtered_MLII,time,samples,start_point),3)
            areas.append(area)
        amps.append(amp)
        offset.append(point_transform_to_origin(right_edge+5,start_point))
        onset.append(point_transform_to_origin(left_edge-5,start_point))
        
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

def find_Q_point(signal,time, R_peaks, time_limit = 0.01,limit=50):
    num_peak = len(R_peaks)
    Q_points = []   
    for i in range(num_peak):
        r_peak = R_peaks[i]
        point = r_peak
        if point-1 >= len(signal):
            
            break
        
        if(signal[point] >= 0 ):
            while point >= R_peaks[i] - limit and signal[point] >= signal[point - 1] or abs(time[r_peak]-time[point]) <= time_limit:             
                point -= 1
                if point >= len(signal):
                    break
        else:
            
            while point >= R_peaks[i] - limit and abs(signal[point]) >= abs(signal[point - 1]) or abs(time[r_peak]-time[point]) <= time_limit:             
                point -= 1
                if point <= len(signal):
                    break
        
        Q_points.append(point)
                        
    return np.asarray(Q_points)

# only works with filtered leads 
def find_S_point(signal,time, R_peaks, time_limit = 0.01, limit=50):
    num_peak = len(R_peaks)
    S_points = []   
    for i in range(num_peak):
        
        r_peak = R_peaks[i]
        point = r_peak
        if point+1 >= len(signal):
           
            break
        
        if(signal[point] >= 0 ):
            while point <= R_peaks[i] + limit and signal[point] >= signal[point + 1] or abs(time[point]-time[r_peak]) <= time_limit:             
                point += 1
                if point >= len(signal):
                   
                    break
        else:
            
            while  point <= R_peaks[i] + limit and abs(signal[point]) >= abs(signal[point + 1]) or abs(time[point]-time[r_peak]) <= time_limit:             
                point += 1
                if point >= len(signal):
                    break
        
        S_points.append(point)
                        
    return np.asarray(S_points) 



def q_s_peak_properties_extractor(patient,time_limit_from_r=0.1,sample_from_point=[5,5], to_area=False,to_savol=True, Order=9,window_len=41, left_limit=50,right_limit=50, distance=1, width=[0,100],plateau_size=[0,100]):
    s_peaks = []
    q_peaks = []
    
    sigs = []
    time = patient.time
    count = 0
    
    heights_q = []
    durations_q = []
    areas_q = []
    onset_q = []
    offset_q = []
    amps_q = []
    promi_q = []
    
    heights_s = []
    durations_s = []
    areas_s = []
    onset_s = []
    offset_s = []
    amps_s = []
    promi_s = []
    
    
    
    print("Patient file: ",patient.filename, "begins")
    
    if(patient.filtered_MLII == []):
        print("Please filter the signal")
        return
    if(patient.segmented_R_pos == []):
        print("please segment the signal to find R peak")
        return
    
    
    
    q_points = find_Q_point(patient.filtered_MLII,patient.time, patient.segmented_R_pos)
    s_points = find_S_point(patient.filtered_MLII,patient.time, patient.segmented_R_pos)
    for r in patient.segmented_R_pos:
        start_point = r-left_limit
        end_point = r+right_limit
        MLII = []
        sig = []
        peak = None 
        properties = None
        height = 0
        time = patient.time[start_point:end_point] 
        if(patient.filtered_MLII[r] >= 0):
            MLII = patient.filtered_MLII
            sig = MLII[start_point:end_point]
            height = min(sig)   
        else:
            MLII = -patient.filtered_MLII
            sig = MLII[start_point:end_point]
            height = min(sig)   
        
        if(to_savol == True):
            sig = savgol_filter(sig,window_len,Order)
            height = min(sig)
            
        peak,properties = peak_properties_extractor(sig,height=height, distance=distance, width=width, plateau_size=plateau_size)

        old_sig = sig
        old_peaks = peak
        origin_peaks = point_transform_to_origin(start_point,peak)
        
        r_range_peaks = []

        for p in range(0,len(old_peaks)):
            if(origin_peaks[p] >= r-5 and origin_peaks[p] <= r+5):
                r_range_peaks.append(old_peaks[p])
                
        sig = sig[r_range_peaks]
        r_peak = 0
        
        if(len(sig) == 0):
            r_peak = origin_to_new_point(start_point,r)

        else: 
            value = max(sig)
            
            index = np.where(sig==value)
        
            index = int(index[0])
    
            r_peak = r_range_peaks[index]
        
        
        peak,properties= peak_properties_extractor(-old_sig,height=height, distance=distance, width=width, plateau_size=plateau_size)

        #do q points 
        
        if(len(peak)==0):
            q_point=q_points[count]
           
            q_peaks.append(q_point)
            durations_q.append(0)
            promi_q.append(0)
            amp = amplitude(patient.filtered_MLII,list(range(q_point-sample_from_point[0],q_point+sample_from_point[1])),0)
            heights_q.append(0)
            if(to_area==True):
                areas_q.append(0)
            amps_q.append(amp)
            offset_q.append(q_point+5)
            onset_q.append(q_point-5)
            #print(q_point)
            
            
            
            s_point=s_points[count]
            
            s_peaks.append(s_point)
            durations_s.append(0)
            promi_s.append(0)
            amp = amplitude(patient.filtered_MLII,list(range(s_point-sample_from_point[0],s_point+sample_from_point[1])),0)
            heights_s.append(0)
            if(to_area==True):
                areas_s.append(0)
            amps_s.append(amp)
            offset_s.append(s_point+5)
            onset_s.append(s_point-5)
            
            count = count+1
            continue
            
        #print("hello len!=0")    
        q_point = 0
        
        previous_point = peak[0]
        temp_point = previous_point
        index = 0
        for i in range(1,len(peak)):
            
            if peak[i] >= r_peak-5:
                break
            if peak[i] > previous_point:
                
                temp_point = peak[i]
                previous_point = peak[i]
                index = index + 1
                
        
        if((patient.time[point_transform_to_origin(start_point,temp_point)]-patient.time[q_points[count]])<=time_limit_from_r and (patient.time[r]-patient.time[point_transform_to_origin(start_point,temp_point)]) >= 0.01 and r>point_transform_to_origin(start_point,temp_point)):
         
            
            q_point = point_transform_to_origin(start_point,temp_point)
            left_ips = np.asarray(properties["left_ips"])
            right_ips = np.asarray(properties["right_ips"])
            left_ips = [int(i) for i in left_ips]
            right_ips = [int(i) for i in right_ips]

        
    
            left_edge = left_ips[index]
            right_edge = right_ips[index]
        
            duration = round(peak_duration(time=patient.time,right_edge=right_edge, left_edge=left_edge,point_from_origin=start_point),3)
            prominences = np.asarray(properties["prominences"])
            prominence = prominences[index]
            height = round(peak_height(-MLII, q_point, prominence,0),3)
      
            durations_q.append(duration)
            promi_q.append(prominence)
            amp = amplitude(patient.filtered_MLII,list(range(q_point-sample_from_point[0],q_point+sample_from_point[1])),0)
        
            heights_q.append(height)
        
            if(to_area==True):
                samples = list(range(left_edge,right_edge+1))
                area = round(area_under_curve(patient.filtered_MLII,patient.time,samples,start_point),3)
                areas_q.append(area)
            
            amps_q.append(amp)
            offset_q.append(point_transform_to_origin(right_edge+5,start_point))
            onset_q.append(point_transform_to_origin(left_edge-5,start_point))
        
            q_peaks.append(q_point)
            #print(q_point)
        else:
            
                    
            q_point=q_points[count]
           
            q_peaks.append(q_point)
            durations_q.append(0)
            promi_q.append(0)
            amp = amplitude(patient.filtered_MLII,list(range(q_point-sample_from_point[0],q_point+sample_from_point[1])),0)
            heights_q.append(0)
            if(to_area==True):
                areas_q.append(0)
            amps_q.append(amp)
            offset_q.append(q_point+5)
            onset_q.append(q_point-5)
            #print(q_point)
 
        #do s points
        s_point = 0
        temp_point = 0
        index = 0
        r_peak = origin_to_new_point(start_point,r)
        for i in range(0,len(peak)):
            
            if peak[i] <= r_peak:
                continue
            
            temp_point = peak[i]
            index = i
            
            break
        
                        
        if((patient.time[point_transform_to_origin(start_point,temp_point)]-patient.time[s_points[count]])<=time_limit_from_r and (patient.time[point_transform_to_origin(start_point,temp_point)]-patient.time[r])>= 0.01 and r<point_transform_to_origin(start_point,temp_point) ):
            
            s_point = point_transform_to_origin(start_point,temp_point)
            
            left_ips = np.asarray(properties["left_ips"])
            right_ips = np.asarray(properties["right_ips"])
            left_ips = [int(i) for i in left_ips]
            right_ips = [int(i) for i in right_ips]

        
            left_edge = left_ips[index]
            right_edge = right_ips[index]
        
            duration = round(peak_duration(time=patient.time,right_edge=right_edge, left_edge=left_edge,point_from_origin=start_point),3)
            prominences = np.asarray(properties["prominences"])
            prominence = prominences[index]
            height = round(peak_height(-MLII, s_point, prominence,0),3)
      
            durations_s.append(duration)
            promi_s.append(prominence)
            amp = amplitude(patient.filtered_MLII,list(range(s_point-sample_from_point[0],s_point+sample_from_point[1])),0)
        
            heights_s.append(height)
        
            if(to_area==True):
                samples = list(range(left_edge,right_edge+1))
                area = round(area_under_curve(patient.filtered_MLII,patient.time,samples,start_point),3)
                areas_s.append(area)
            
            amps_s.append(amp)
            offset_s.append(point_transform_to_origin(right_edge+5,start_point))
            onset_s.append(point_transform_to_origin(left_edge-5,start_point))
            s_peaks.append(s_point)
                
        else:
            s_point=s_points[count]
            
            s_peaks.append(s_point)
            durations_s.append(0)
            promi_s.append(0)
            amp = amplitude(patient.filtered_MLII,list(range(s_point-sample_from_point[0],s_point+sample_from_point[1])),0)
            heights_s.append(0)
            if(to_area==True):
                areas_s.append(0)
            amps_s.append(amp)
            offset_s.append(s_point+5)
            onset_s.append(s_point-5)
         
        count = count+1
            
                

        
    q_properties = {
        "peaks" : q_peaks,
        "durations" : durations_q,
        "prominences" : promi_q,
        "height" : heights_q,
        "amplitudes" : amps_q,
        "areas" : areas_q,
        "onset" : onset_q,
        "offset" : offset_q
    }
    
    s_properties = {
        "peaks" : s_peaks,
        "durations" : durations_s,
        "prominences" : promi_s,
        "height" : heights_s,
        "amplitudes" : amps_s,
        "areas" : areas_s,
        "onset" : onset_s,
        "offset" : offset_s
    }
    
   
    return   q_peaks, q_properties , s_peaks, s_properties
    





def find_index(ls,value):
    index = np.where(ls==value)
        
    index = int(index[0])
    
    return index


def find_values_in_properties(patient,signal ,peak, properties, index, sample_from_point, start_point,to_area):

    point = point_transform_to_origin(peak,start_point)
            
    left_ips = np.asarray(properties["left_ips"])
    right_ips = np.asarray(properties["right_ips"])
    left_ips = [int(i) for i in left_ips]
    right_ips = [int(i) for i in right_ips]

        
    
    left_edge = left_ips[index]
    right_edge = right_ips[index]
        
    duration = round(peak_duration(time=patient.time,right_edge=right_edge, left_edge=left_edge,point_from_origin=start_point),3)
    prominences = np.asarray(properties["prominences"])
    prominence = prominences[index]
    height = round(peak_height(signal, point, prominence,0),3)
      
    
    amp = amplitude(patient.filtered_MLII,list(range(point-sample_from_point[0],point+sample_from_point[1])),0)
        
    area = None
        
    if(to_area==True):
        samples = list(range(left_edge,right_edge+1))
        area = round(area_under_curve(patient.filtered_MLII,patient.time,samples,start_point),3)

            
    
    offset = point_transform_to_origin(right_edge+5,start_point)
    onset = point_transform_to_origin(left_edge-5,start_point)
    
    return point, duration, prominence, height, amp, area, offset, onset


def euclidean_distance(x,y):
 
    return sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))

    
def sudo_k_mean(ls, time, amp):
    first_element = ls[0]
    last_element = ls[len(ls)-1]
    
    left = []
    right = []
    
 
    left.append(first_element)
    right.append(last_element)
    for l in range(1, len(ls)-1):
        time_1 = [time[i] for i in left]
        time_2 = [time[i] for i in right]
        amp_1 = [amp[i] for i in left]
        amp_2 = [amp[i] for i in right]
       
    
        
        centroid_1_x = average(time_1)
        centroid_1_y = average(amp_1)
       # print(centroid_1, "centroid_1")
        centroid_2_x = average(time_2)
        centroid_2_y = average(amp_2)

       # print(centroid_2, "centroid_2")
       
        point = ls[l]
        time_point = time[point]
        amp_point = amp[point]
       # print(point, "point")
       # print("time", time_point)
        
        diff_1 = euclidean_distance([centroid_1_x, centroid_1_y],[time_point,amp_point])
        diff_2 = euclidean_distance([centroid_2_x, centroid_2_y],[time_point,amp_point])
        
        if(diff_1 > diff_2):
            right.append(point)
        else:
            left.append(point)
        
    return left, right 
        
        
def highest_peak(peaks, signal):
    signal = signal[peaks]
    max_signal = max(signal)
   
    index = np.where(signal==max_signal)
    
    highest = 0
    
    
    if(len(index[0]) > 1):
        first = index[0][0]
        last = index[0][len(index[0])-1]
        index = round((first+last)/2,0)
        index = int(index)
        
    else:
        index = int(index[0])
    
    highest = peaks[index]
    
    
    return highest


 

def p_and_t_peak_properties_extractor(patient,time_limit_from_r=0.1,sample_from_point=[5,5], to_area=False,to_savol=False, Order=9,window_len=31, left_limit=50,right_limit=50, distance=1, width=[0,100],plateau_size=[0,100]):
    p_peaks = []
    p_heights = []
    p_durations = []
    p_areas = []
    p_onset = []
    p_offset = []
    p_amps = []
    p_promi = []
    sigs = []
    
    t_peaks = []
    t_heights = []
    t_durations = []
    t_areas = []
    t_onset = []
    t_offset = []
    t_amps = []
    t_promi = []
    
    p_positives = []
    p_negatives = []
    t_positives = []
    t_negatives = []
    
    
    time = patient.time
    count = 0
    print("Patient file: ",patient.filename, "begins")
    
    if(patient.filtered_MLII == []):
        print("Please filter the signal")
        return
    if(patient.segmented_R_pos == []):
        print("please segment the signal to find R peak")
        return
    
    
    

    r_peaks = patient.segmented_R_pos
    q_peaks = find_Q_point(patient.filtered_MLII,patient.time, r_peaks, time_limit = 0.01,limit=50)
    s_peaks = find_S_point(patient.filtered_MLII,patient.time, r_peaks, time_limit = 0.01,limit=50)
    #q_peaks = patient.Q_points_properites["peaks"]
    #s_peaks = patient.S_points_properites["peaks"]
        
    first_r_sig = patient.filtered_MLII[q_peaks[0]-100:q_peaks[0]]
    last_r_sig = patient.filtered_MLII[s_peaks[len(s_peaks)-1]:s_peaks[len(s_peaks)-1]+100]
    
    pre_r_sig = first_r_sig
    start_pre_r = q_peaks[0]-100
    end_pre_r = q_peaks[0]
    post_r_sig = patient.filtered_MLII[s_peaks[0]:r_peaks[1]]
    start_post_r = s_peaks[0]
    end_post_r = r_peaks[1]
    
    for i in range(0,len(r_peaks)):
        ####pre_processing
        
        negative_pre = -pre_r_sig
        
        
        if(to_savol == True):
            pre_r_sig = savgol_filter(pre_r_sig,window_len,Order)
            negative_pre = savgol_filter(negative_pre,window_len,Order)

        peak,properties= peak_properties_extractor(pre_r_sig, distance=distance, width=width, plateau_size=plateau_size)
        neg_peak,neg_properties= peak_properties_extractor(negative_pre, distance=distance, width=width, plateau_size=plateau_size)
        #print(len(neg_peak),i,r_peaks[i], len(negative_pre))
        ########
        #######do operation to find the p wave
        
        abs_peak=[]
        abs_neg_peak = []
       
        
        point = 0
        duration = 0
        prominence = 0
        height = 0
        amp=0
        area=0
        offset=0
        onset = 0        
        point_neg=0
        duration_neg=0
        prominence_neg=0
        height_neg=0
        amp_neg=0
        area_neg=0
        offset_neg=0
        onset_neg = 0
        left=0
        right= 0
        p_pos=0
        p_neg=0
        
        #print(len(abs_peak), "hi")
        if(len(peak) == 0 ):
           
            abs_peak=list(range(start_pre_r,end_pre_r))
            if(len(abs_peak) == 0):
                p_pos = round((start_pre_r+end_pre_r)/2,0)
            
            else:
                left, right= sudo_k_mean(abs_peak, time, patient.filtered_MLII)
                p_pos = highest_peak(right, patient.filtered_MLII)
            
            p_positives.append(p_pos)
            
        
        else:
            
            abs_peak = [point_transform_to_origin(p, start_pre_r) for p in peak]
            left, right= sudo_k_mean(abs_peak, time, patient.filtered_MLII)
            p_pos = highest_peak(right, patient.filtered_MLII)
            p_positives.append(p_pos)
            index_pos = find_index(abs_peak, p_pos)
            p_peak = peak[index_pos]
            point, duration, prominence, height, amp, area, offset, onset=find_values_in_properties(patient,patient.filtered_MLII ,p_peak, properties, index_pos, sample_from_point, start_pre_r,to_area)


        
        if(len(neg_peak) == 0 ):
            
            
            abs_neg_peak=list(range(start_pre_r,end_pre_r))
            if(len(abs_neg_peak) == 0):
                p_neg = round((start_pre_r+end_pre_r)/2,0)
            
            else:
                neg_left, neg_right = sudo_k_mean(abs_neg_peak, time, -patient.filtered_MLII)
                p_neg = highest_peak(neg_right, patient.filtered_MLII)
            p_negatives.append(p_neg)
        
        else:
            abs_neg_peak = [point_transform_to_origin(p, start_pre_r) for p in neg_peak]
            neg_left, neg_right = sudo_k_mean(abs_neg_peak, time, -patient.filtered_MLII)
            
            p_neg = highest_peak(neg_right,-patient.filtered_MLII)
            p_negatives.append(p_neg)
            index_neg = find_index(abs_neg_peak, p_neg)
            p_neg_peak = neg_peak[index_neg]
            point_neg, duration_neg, prominence_neg, height_neg, amp_neg, area_neg, offset_neg, onset_neg=find_values_in_properties(patient,-patient.filtered_MLII ,p_neg_peak, neg_properties, index_neg, sample_from_point, start_pre_r,to_area)

        
        
        ######Turn to normal peak to find the other properties      
            
            
        p_peaks.append((p_positives,p_negatives))
        p_heights.append((height,height_neg))
        p_durations.append((duration,duration_neg))
        p_areas.append((area,area_neg))
        p_onset.append((onset,onset_neg))
        p_offset.append((offset,offset_neg))
        p_amps.append((amp, amp_neg))
        p_promi.append((prominence,prominence_neg))
        
        ##################################################
        negative_post = -post_r_sig
        
        if(to_savol == True):
            post_r_sig = savgol_filter(post_r_sig,window_len,Order)
            negative_post = savgol_filter(negative_post,window_len,Order)

        peak,properties= peak_properties_extractor(post_r_sig, distance=distance, width=width, plateau_size=plateau_size)
        neg_peak,neg_properties= peak_properties_extractor(negative_post, distance=distance, width=width, plateau_size=plateau_size)
        
        ########
        #######do operation to find the t wave
        
        abs_peak=[]
        abs_neg_peak = []
       
        
        point = 0
        duration = 0
        prominence = 0
        height = 0
        amp=0
        area=0
        offset=0
        onset = 0        
        point_neg=0
        duration_neg=0
        prominence_neg=0
        height_neg=0
        amp_neg=0
        area_neg=0
        offset_neg=0
        onset_neg = 0
        left=0
        right= 0
        p_pos=0
        p_neg=0
        
        
        #print(len(abs_peak), "hi")
        if(len(peak) == 0 ):
            
            abs_peak=list(range(start_post_r,end_post_r))
            
            if(len(abs_peak) == 0):
                t_pos = round((start_post_r+end_post_r)/2,0)
            
            else:
                
                left, right= sudo_k_mean(abs_peak, time, patient.filtered_MLII)
                t_pos = highest_peak(left, patient.filtered_MLII)
                
           
            
            
            t_positives.append(t_pos)
            
        
        else:
            abs_peak = [point_transform_to_origin(p, start_post_r) for p in peak]
            left, right= sudo_k_mean(abs_peak, time, patient.filtered_MLII)
            t_pos = highest_peak(left, patient.filtered_MLII)
            t_positives.append(t_pos)
            index_pos = find_index(abs_peak, t_pos)
            t_peak = peak[index_pos]
            point, duration, prominence, height, amp, area, offset, onset=find_values_in_properties(patient,patient.filtered_MLII ,t_peak, properties, index_pos, sample_from_point, start_pre_r,to_area)


        
        if(len(neg_peak) == 0 ):
            
            abs_neg_peak=list(range(start_post_r,end_post_r))
           
            if(len(abs_neg_peak) == 0):
                
                t_neg = round((start_post_r+end_post_r)/2,0)
            else:
                neg_left, neg_right= sudo_k_mean(abs_neg_peak, time, -patient.filtered_MLII)
                t_neg = highest_peak(neg_left, patient.filtered_MLII)
            t_negatives.append(t_neg)
        
        else:
            abs_neg_peak = [point_transform_to_origin(p, start_post_r) for p in neg_peak]
            neg_left, neg_right = sudo_k_mean(abs_neg_peak, time, -patient.filtered_MLII)
            t_neg = highest_peak(neg_left,-patient.filtered_MLII)
            t_negatives.append(t_neg)
            index_neg = find_index(abs_neg_peak, t_neg)
            t_neg_peak = neg_peak[index_neg]
            point_neg, duration_neg, prominence_neg, height_neg, amp_neg, area_neg, offset_neg, onset_neg=find_values_in_properties(patient,-patient.filtered_MLII ,t_neg_peak, neg_properties, index_neg, sample_from_point, start_pre_r,to_area)  
        

        t_peaks.append((t_positives,t_negatives))
        t_heights.append((height,height_neg))
        t_durations.append((duration,duration_neg))
        t_areas.append((area,area_neg))
        t_onset.append((onset,onset_neg))
        t_offset.append((offset,offset_neg))
        t_amps.append((amp, amp_neg))
        t_promi.append((prominence,prominence_neg))
        
        ########next wave _________________________________________
        
        if(i == len(r_peaks)-1):
            break
           
        pre_r_sig = patient.filtered_MLII[s_peaks[i]:q_peaks[i+1]]
        
       # print("before next ",patient.filtered_MLII[s_peaks[i]:q_peaks[i-1]])
        start_pre_r = s_peaks[i]
        end_pre_r = q_peaks[i+1]
        if(i == len(r_peaks)-2):
            post_r_sig = last_r_sig
            start_post_r = s_peaks[len(s_peaks)-1]
            end_post_r = s_peaks[len(s_peaks)-1]+100
            
        else:
            post_r_sig = patient.filtered_MLII[s_peaks[i+1]:q_peaks[i+2]]
            start_post_r = s_peaks[i+1]
            end_post_r = q_peaks[i+2]
            
    
    p_properties = {
        "peaks" : p_peaks,
        "durations" : p_durations,
        "prominences" : p_promi,
        "height" : p_heights,
        "amplitudes" : p_amps,
        "areas" : p_areas,
        "onset" : p_onset,
        "offset" : p_offset
    }
    
    t_properties = {
        "peaks" : t_peaks,
        "durations" : t_durations,
        "prominences" : t_promi,
        "height" : t_heights,
        "amplitudes" : t_amps,
        "areas" : t_areas,
        "onset" : t_onset,
        "offset" : t_offset
    }
        
    return p_positives, p_negatives, p_properties, t_positives, t_negatives, t_properties






