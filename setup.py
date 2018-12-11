from codes.python import load_mitdb,ECG_denoising
from codes.python import simple_heartbeat_segmentation as shs


mit100 = load_mitdb.load_patient_record("mitdb","100")
load_mitdb.display_signal_in_seconds(mit100,mit100.MLII,3)
filter = ECG_denoising.ECG_FIR_filter()
mit100.filtered_MLII = ECG_denoising.denoising_signal_FIR(mit100.MLII,filter)
mit100.filtered_V1 = ECG_denoising.denoising_signal_FIR(mit100.V1,filter)
mit100.beat_1, mit100.class_ID, mit100.R_pos = shs.segment_beat(mit100.filtered_MLII, mit100.annotations, 90, 90)
#load_mitdb.display_signal_in_seconds(mit100,mit100.filtered_MLII,3)

beats = mit100.beat_1

for b in beats:
    load_mitdb.display_signal(b)



