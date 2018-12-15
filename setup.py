from codes.python import load_mitdb,ECG_denoising
from codes.python import simple_heartbeat_segmentation as shs
import numpy as np

mitdb = load_mitdb.load_mitdb()
mit100 = load_mitdb.load_patient_record("mitdb","100")
#load_mitdb.display_signal_in_seconds(mit100,mit100.MLII,3)
filter_ecg = ECG_denoising.ECG_FIR_filter()
mit100.filtered_MLII = ECG_denoising.denoising_signal_FIR(mit100.MLII,filter_ecg)
mit100.filtered_V1 = ECG_denoising.denoising_signal_FIR(mit100.V1,filter_ecg)
mit100.segmented_beat_1, mit100.segmented_class_ID, mit100.segmented_beat_class, mit100.segmented_R_pos, mit100.segmented_valid_R, mit100.segmented_original_R  = shs.segment_beat(mit100.filtered_MLII, mit100.time, mit100.annotations, 90, 90)
#np100R = np.concatenate((mit100.segmented_original_R,mit100.segmented_R_pos,mit100.segmented_valid_R ))

MLII = mit100.MLII
np100_MLII = np.array(mit100.segmented_beat_1)

np100R = np.array(mit100.segmented_original_R)
np_r_poses = np.array(mit100.segmented_R_pos)
np_valid_r = np.array(mit100.segmented_valid_R)
#np100R = np.append(np100R, np_r_poses, axis=1)
np100R = np.column_stack((np100R, np_r_poses))
np100R = np.column_stack((np100R, np_valid_r))


count = 0
for i in range(0,len(np100R)):
    ori_R =np100R[i,1]
    seg_R = np100R[i,3]
    if(ori_R <= seg_R):
        count = count+ 1

percent = (len(np100R)/count) * 100
       
    
    


np100R.shape
np100R



#load_mitdb.display_signal_in_seconds(mit100,mit100.filtered_MLII,3)








