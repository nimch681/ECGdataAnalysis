import numpy as np 
import math
import matplotlib.pyplot as plt
from numpy import array
from codes.python import QRS_detector
import sys
import csv
import os
import operator
import matplotlib.pyplot as plt
import wfdb
mit100 = load_mitdb.load_patient_record("mitdb","100")
filter_ecg = ECG_denoising.ECG_FIR_filter()
mit100.filtered_MLII = ECG_denoising.denoising_signal_FIR(mit100.MLII,filter_ecg)
mit100.filtered_V1 = ECG_denoising.denoising_signal_FIR(mit100.V1,filter_ecg)
#mit100.segmented_beat_1, mit100.segmented_class_ID, mit100.segmented_beat_class, mit100.segmented_R_pos = shs.segment_beat(mit100.filtered_MLII, mit100.time, mit100.annotations, 90, 90)
beats = mit100.beat_1

beat_limits = []

for i in range(0, len(beats)):
    limit_1 = beats[i][0][0]
    limit_2 = beats[i][0][len(beats[i][0])-1]
    beat_limits.append((limit_1,limit_2))

beat_time_limits = []

for i in range(0, len(beats)):
    limit_1 = beats[i][1][0]
    limit_2 = beats[i][1][len(beats[i][1])-1]
    beat_time_limits.append((limit_1,limit_2))
    

#mit100.beat_1[0][0:len(mit100.beat_1[0])]
#numbers = list(range(1, 1000))
#numbers
#len(mit100.beat_1[0])
findpeaks_limit = 0.35
findpeaks_spacing = 72#360
integration_window = 16

filtered_ecg_measurements = None
differentiated_ecg_measurements = None
squared_ecg_measurements = None
#integrated_ecg_measurements = None
detected_peaks_indices = None #important
detected_peaks_values = None # important

def detect_peaks(signal):
    """
    Method responsible for extracting peaks from loaded ECG measurements data through measurements processing.
    """
    # Extract measurements from loaded ECG data.
    ecg_measurements = signal

    # Measurements filtering - 0-15 Hz band pass filter.
    #self.filtered_ecg_measurements = self.bandpass_filter(ecg_measurements, lowcut=self.filter_lowcut,
                                                            #highcut=self.filter_highcut, signal_freq=self.signal_frequency,
                                                            #filter_order=self.filter_order)
    #self.filtered_ecg_measurements[:5] = self.filtered_ecg_measurements[5]

        # Derivative - provides QRS slope information.
    differentiated_ecg_measurements = np.ediff1d(ecg_measurements)

        # Squaring - intensifies values received in derivative.
    squared_ecg_measurements = ecg_measurements ** 2

        # Moving-window integration.
    integrated_ecg_measurements = np.convolve(squared_ecg_measurements, np.ones(integration_window))

    

        # Fiducial mark - peak detection on integrated measurements.
    detected_peaks_indices = findpeaks(data=integrated_ecg_measurements,
                                                     limit=findpeaks_limit,
                                                     spacing=findpeaks_spacing)

    detected_peaks_values = integrated_ecg_measurements[detected_peaks_indices]

    return detected_peaks_indices, detected_peaks_values

def findpeaks(data, spacing=1, limit=None):
    """
        Janko Slavic peak detection algorithm and implementation.
        https://github.com/jankoslavic/py-tools/tree/master/findpeaks
        Finds peaks in `data` which are of `spacing` width and >=`limit`.
        :param ndarray data: data
        :param float spacing: minimum spacing to the next peak (should be 1 or more)
        :param float limit: peaks should have value greater or equal
        :return array: detected peaks indexes array
    """
    len = data.size
    x = np.zeros(len + 2 * spacing)
    x[:spacing] = data[0] - 1.e-6
    x[-spacing:] = data[-1] - 1.e-6
    x[spacing:spacing + len] = data
    peak_candidate = np.zeros(len)
    peak_candidate[:] = True
    for s in range(spacing):
        start = spacing - s - 1
        h_b = x[start: start + len]  # before
        start = spacing
        h_c = x[start: start + len]  # central
        start = spacing + s + 1
        h_a = x[start: start + len]  # after
        peak_candidate = np.logical_and(peak_candidate, np.logical_and(h_c > h_b, h_c > h_a))

    ind = np.argwhere(peak_candidate)
    ind = ind.reshape(ind.size)
    if limit is not None:
        ind = ind[data[ind] > limit]
    return ind

def detect_qrs(self):
    """
    Method responsible for classifying detected ECG measurements peaks either as noise or as QRS complex (heart beat).
    """
    for detected_peak_index, detected_peaks_value in zip(self.detected_peaks_indices, self.detected_peaks_values):

        try:
            last_qrs_index = self.qrs_peaks_indices[-1]
        except IndexError:
            last_qrs_index = 0

        # After a valid QRS complex detection, there is a 200 ms refractory period before next one can be detected.
        if detected_peak_index - last_qrs_index > self.refractory_period or not self.qrs_peaks_indices.size:
            # Peak must be classified either as a noise peak or a QRS peak.
            # To be classified as a QRS peak it must exceed dynamically set threshold value.
            if detected_peaks_value > self.threshold_value:
                self.qrs_peaks_indices = np.append(self.qrs_peaks_indices, detected_peak_index)

                # Adjust QRS peak value used later for setting QRS-noise threshold.
                self.qrs_peak_value = self.qrs_peak_filtering_factor * detected_peaks_value + \
                                        (1 - self.qrs_peak_filtering_factor) * self.qrs_peak_value
            else:
                self.noise_peaks_indices = np.append(self.noise_peaks_indices, detected_peak_index)

                # Adjust noise peak value used later for setting QRS-noise threshold.
                self.noise_peak_value = self.noise_peak_filtering_factor * detected_peaks_value + \
                                        (1 - self.noise_peak_filtering_factor) * self.noise_peak_value

            # Adjust QRS-noise threshold value based on previously detected QRS or noise peaks value.
            self.threshold_value = self.noise_peak_value + \
                                    self.qrs_noise_diff_weight * (self.qrs_peak_value - self.noise_peak_value)

    # Create array containing both input ECG measurements data and QRS detection indication column.
    # We mark QRS detection with '1' flag in 'qrs_detected' log column ('0' otherwise).
    measurement_qrs_detection_flag = np.zeros([len(self.ecg_data), 1])
    measurement_qrs_detection_flag[self.qrs_peaks_indices] = 1
    self.ecg_data_detected = np.append(self.ecg_data, measurement_qrs_detection_flag)