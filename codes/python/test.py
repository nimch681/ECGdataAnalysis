import numpy as np 
import math
import matplotlib.pyplot as plt
from numpy import array
from codes.python import QRS_detector
from codes.python import QRSDetectorOffline
import os


np100 = array([mit100.MLII])
np100T = np100.T

qrs_detector = QRS_detector.QRSDetectorOffline(ecg_data_raw = mit100.MLII, fs = 360, verbose=True, plot_data=True, show_plot=True)

qrs_detector = QRSDetectorOffline.QRSDetectorOffline(ecg_data_path= "ecg_data_1.csv", verbose=True, log_data=True, plot_data=True, show_plot=True)
