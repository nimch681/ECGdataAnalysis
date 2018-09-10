import os
import pandas as pd
from tabulate import tabulate as tb
from sklearn.decomposition import PCA
import numpy as np

MAINPATH = 'C:\\Users\\nimch681\\AppData\Loca  l\\GitHubDesktop\\app-1.3.5\\ECGdataAnalysis'
os.chdir(MAINPATH)
DATAPATH = 'database/UCIarrhythmia.csv'
ucidata = pd.read_csv(DATAPATH)
ucidata = ucidata.values
ucimatrix = np.asmatrix(ucidata)
pca = PCA(n_components=2)

X=np.copy(ucidata[:,:279]) #getting the features values from uci data 
y=np.copy(ucidata[:,279]) #getting the class values from uci data


 
ucidata