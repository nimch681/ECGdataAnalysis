from codes.python import load_mitdb
from codes.python import ECG_denoising


mit100 = load_mitdb.load_patient_record("mitdb","100")
load_mitdb.display_signal_in_seconds(mit100,mit100.MLII,3)
filter = ECG_denoising.ECG_FIR_filter()
ECG_denoising.denoising_signal_FIR(mit100,filter)
load_mitdb.display_signal_in_seconds(mit100,mit100.MLII,3)
