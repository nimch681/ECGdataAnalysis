from __future__ import print_function
import numpy as np
from scipy import signal
from scipy.signal import savgol_filter
import operator
from numpy import array
from wfdb import processing, plot
from sklearn import metrics
import wfdb
from math import *
import sys


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

##### Interval extractors


def average(numbers):
    return float(sum(numbers)) / len(numbers)    

def point_transform_to_origin(por,point):
    point_from_origin = por + point 
    return point_from_origin

def origin_to_new_point(por,point_from_origin):
    point = point_from_origin - por
    return point

def sub_signal_interval(time, start_point, end_point,point_from_origin=0):
    start_point = point_transform_to_origin(point_from_origin,start_point)
    end_point = point_transform_to_origin(point_from_origin,end_point)
    
    return float(abs(time[end_point]-time[start_point]))

def average_by_sample(interval,to_ten=True,to_fifty=False, to_all=True):
     
    ten = []
    fifty = []
    all_avg = []
    num_max_beat = len(interval)
    
    
    if to_all == True:
        average_all = global_average(interval)
        all_avg = [average_all] * num_max_beat

    for i in range(0,num_max_beat):
        average_10 = 0
        average_50 = 0
        
        if to_ten == True:
        
            average_10=local_average(interval,num_max_beat, i, 10)
            
            ten.append(average_10)
                
        if to_fifty == True:
            
            average_50=local_average(interval,num_max_beat, i, 50)
            
            fifty.append(average_50)

        
    return ten, fifty, all_avg   

def global_average(interval):
    return average(interval)

def local_average(interval,len_of_list, pos, average_for_calulation):
    half_ave = int(average_for_calulation/2)
    ave = 0
    if pos >= len_of_list-(half_ave+1):
        ave = average(interval[pos-half_ave:len_of_list-1])
    if pos <= half_ave:
        ave = average(interval[1:pos+(half_ave+1)])
    if (pos > half_ave and pos < len_of_list-(half_ave+1)):
        ave = average(interval[pos-half_ave:pos+(half_ave+1)])
    
    return ave

def interval_and_average(record, ten=True,fifty=True, all_avg=False):

    
    if(record.Q_points == []):
        eprint("please extract signal to find q properties")
        return
    if(record.S_points == []):
        eprint("please extract signal to find s properties")
        return
    if(record.P_points == []):
        eprint("please extract signal to find P properties")
        return
    if(record.T_points == []):
        eprint("please extract signal to find T properties")
        return
    
    QRS = []

    P_Q = []
    P_Q_neg = []

    P_R = []
    P_R_neg = []

    S_T = []
    S_T_neg = []

    R_T = []
    R_T_ten = []
    R_T_fifty = []
    R_T_all = []
    R_T_neg = []

    P_T = []
    neg_P_T = []
    P_T_neg = []
    neg_P_T_neg = []


    for i in range(0, len(record.Q_points)):

        qrs = sub_signal_interval(record.time,int(record.Q_points_properites["onset"][i]),int(record.S_points_properites["offset"][i]),0)
        QRS.append(qrs)

        p_q = sub_signal_interval(record.time,int(record.P_points_properites["onset"][i][0]),int(record.Q_points_properites["offset"][i]),0)
        P_Q.append(p_q)

        p_q_neg = sub_signal_interval(record.time,int(record.P_points_properites["onset"][i][1]),int(record.Q_points_properites["offset"][i]),0)
        P_Q_neg.append(p_q_neg)
        
        p_r = sub_signal_interval(record.time,int(record.P_points_properites["onset"][i][0]),int(record.R_pos_properites["offset"][i]),0)
        P_R.append(p_r)

        p_r_neg = sub_signal_interval(record.time,int(record.P_points_properites["onset"][i][1]),int(record.R_pos_properites["offset"][i]),0)
        P_R_neg.append(p_r_neg) 

        s_t = sub_signal_interval(record.time,int(record.S_points_properites["onset"][i]),int(record.T_points_properites["offset"][i][0]),0)
        S_T.append(s_t)

        s_t_neg = sub_signal_interval(record.time,int(record.S_points_properites["onset"][i]),int(record.T_points_properites["offset"][i][1]),0)
        S_T_neg.append(s_t_neg)

        r_t = sub_signal_interval(record.time,int(record.R_pos_properites["onset"][i]),int(record.T_points_properites["offset"][i][0]),0)
        R_T.append(r_t)

        r_t_neg = sub_signal_interval(record.time,int(record.R_pos_properites["onset"][i]),int(record.T_points_properites["offset"][i][1]),0)
        R_T_neg.append(r_t_neg)

        p_t = sub_signal_interval(record.time,int(record.P_points_properites["onset"][i][0]),int(record.T_points_properites["offset"][i][0]),0)
        P_T.append(p_t)

        n_p_t = sub_signal_interval(record.time,int(record.P_points_properites["onset"][i][1]),int(record.T_points_properites["offset"][i][0]),0)
        neg_P_T.append(n_p_t)

        p_t_n = sub_signal_interval(record.time,int(record.P_points_properites["onset"][i][0]),int(record.T_points_properites["offset"][i][1]),0)
        P_T_neg.append(p_t_n)

        n_p_t_n = sub_signal_interval(record.time,int(record.P_points_properites["onset"][i][1]),int(record.T_points_properites["offset"][i][1]),0)
        neg_P_T_neg.append(n_p_t_n)


        
    
    QRS_ten, QRS_fifty, QRS_all = average_by_sample(QRS,ten,fifty, all_avg)
    p_q_ten, p_q_fifty, p_q_all = average_by_sample(P_Q,ten,fifty, all_avg)
    p_r_ten, p_r_fifty, p_r_all = average_by_sample(P_R,ten,fifty, all_avg)
    s_t_ten, s_t_fifty, s_t_all = average_by_sample(S_T,ten,fifty, all_avg)
    r_t_ten, r_t_fifty, r_t_all = average_by_sample(R_T,ten,fifty, all_avg)
    p_t_ten, p_t_fifty, p_t_all = average_by_sample(P_T,ten,fifty, all_avg)
    
    #print(len(QRS_ten))


    QRS_properties = {
        "interval" : QRS,
        "average_ten" : QRS_ten,
        "average_fifty" : QRS_fifty,
        "average_all" : QRS_all
       
    }

    P_Q_properties = {
        "interval" : P_Q,
        "average_ten" : p_q_ten,
        "average_fifty" : p_q_fifty,
        "average_all" : p_q_all
       
    }


    P_R_properties = {
        "interval" : P_R,
        "average_ten" : p_r_ten,
        "average_fifty" : p_r_fifty,
        "average_all" : p_r_all
       
    }

    S_T_properties = {
        "interval" : S_T,
        "average_ten" : s_t_ten,
        "average_fifty" : s_t_fifty,
        "average_all" : s_t_all
       
    }

    R_T_properties = {
        "interval" : R_T,
        "average_ten" : r_t_ten,
        "average_fifty" : r_t_fifty,
        "average_all" : r_t_all
       
    }

    P_T_properties = {
        "interval" : P_T,
        "average_ten" : p_t_ten,
        "average_fifty" : p_t_fifty,
        "average_all" : p_t_all
       
    }

  
    return QRS_properties, P_Q_properties, P_Q_neg, P_R_properties, P_R_neg, S_T_properties, S_T_neg,R_T_properties, R_T_neg, P_T_properties,neg_P_T, P_T_neg, neg_P_T_neg