import numpy as np
from scipy import signal
from scipy.signal import savgol_filter
import operator
from numpy import array
from wfdb import processing, plot
from sklearn import metrics
import wfdb
from math import *



##### Interval extractors



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