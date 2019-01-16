#!/usr/bin/env python

"""
simple_heartbeat_segmentation.py
Description:
contains code for simple heartbeat segmentation
Borrowed and slighly changed the code from VARPA, University of Coruna: Mondejar Guerra, Victor M.
24 Oct 2017
"""
import operator
import numpy as np

def segment_beat(signal,time,annotations, winL, winR):
    class_ID = []
    beat = []
    R_poses = []
    beat_class = []
    Original_R_poses = []  
    valid_R = []
    size_RR_max = 22
    pos = 0
    MITBIH_classes = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F']#, 'P', '/', 'f', 'u']
    AAMI_classes = []
    AAMI_classes.append(['N', 'L', 'R'])                    # N
    AAMI_classes.append(['A', 'a', 'J', 'S', 'e', 'j'])     # SVEB 
    AAMI_classes.append(['V', 'E'])                         # VEB
    AAMI_classes.append(['F'])       


    for a in annotations:
    
        aS = a.split()
            
        pos = int(aS[1])
        if(pos > len(signal)):
            break
            
        originalPos = int(aS[1])
        classAnttd = str(aS[2])
    #originalPos = int(aS[1])
   
        if pos > size_RR_max and pos < (len(signal) - size_RR_max):
            index, value = max(enumerate(signal[pos - size_RR_max : pos + size_RR_max]), key=operator.itemgetter(1))
            pos = (pos - size_RR_max) + index
            
   
        
        beat_poses = list(range(pos - winL, pos + winR))
        beat_poses = [int(i) for i in beat_poses]
        zero = np.zeros(winL+winR)

        if classAnttd in MITBIH_classes:
        
            if(pos > winL and pos < (len(signal) - winR)):

                
                beat.append((beat_poses,time[pos - winL : pos + winR],signal[pos - winL : pos + winR]))
           
                for i in range(0,len(AAMI_classes)):
                    if classAnttd in AAMI_classes[i]:
                        class_AAMI = i
                        break #exit loop
                #convert class
                class_ID.append(class_AAMI)

                valid_R.append(1)
               
            else:
               valid_R.append(0)
               class_ID.append(-1)
               beat.append((beat_poses,zero,zero))

               #if(pos > winL and pos < (len(signal) - winR)):
              
        else:
            valid_R.append(0)
            class_ID.append(-1)
            #if(pos > winL and pos < (len(signal) - winR)):
            beat.append((beat_poses,zero,zero))


        R_poses.append(pos)
        beat_class.append(classAnttd)
        Original_R_poses.append((originalPos)
        
    
        #Original_R_poses = np.append(Original_R_poses, originalPos)
    return beat, class_ID, beat_class, R_poses,valid_R, Original_R_poses
    