from codes.python import load_mitdb,ECG_denoising
from codes.python import simple_heartbeat_segmentation as shs


mit100 = load_mitdb.load_patient_record("mitdb","100")
load_mitdb.display_signal_in_seconds(mit100,mit100.MLII,3)
filter_ecg = ECG_denoising.ECG_FIR_filter()
mit100.filtered_MLII = ECG_denoising.denoising_signal_FIR(mit100.MLII,filter_ecg)
mit100.filtered_V1 = ECG_denoising.denoising_signal_FIR(mit100.V1,filter_ecg)
#mit100.segmented_beat_1, mit100.segmented_class_ID, mit100.segmented_beat_class, mit100.segmented_R_pos = shs.segment_beat(mit100.filtered_MLII, mit100.time, mit100.annotations, 90, 90)
#load_mitdb.display_signal_in_seconds(mit100,mit100.filtered_MLII,3)







