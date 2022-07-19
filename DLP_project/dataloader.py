import pandas as pd
from torchvision import transforms,models
from torch.utils import data
from PIL import Image
import numpy as np
import os
from io import BytesIO
import numpy
from DB_CRUD import DB

import os
import sys
import numpy
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime
from scipy.signal import butter, filtfilt, resample
import torch
from torch import device
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn import preprocessing

def blob_to_array(text):
    out = BytesIO(text)
    out.seek(0)
    npz = numpy.load(out,allow_pickle=True)
    arr = npz['offsets']
    npz.close()
    return arr

normT1 = torch.nn.InstanceNorm1d(1)
nyq = 0.5 * 25
low = 0.5 / nyq
high = 8 / nyq
b, a = butter(2, [low, high], btype='band')

def butter_bandpass_filter(data):
    y = filtfilt(b, a, data)
    return y

def get_features(data):
    seg = data.reshape(1, -1)
    seg = resample(seg, 750, axis=1) #降到剩30s*25HZ=750點
    seg = butter_bandpass_filter(seg)

    normalizer = preprocessing.Normalizer().fit(seg)  #擬合原始資料，data是多維陣列
    normalizer.transform(seg) #正則化

    return seg

class SimulaterDataset(data.Dataset):
   
    def __init__(self, mode):

      
        self.mode = mode

        self.datasTrain = []
        self.labelsTrain = []
        self.dataIdTrain = []

        self.datasTest = []
        self.labelsTest = []
        self.dataIdTest = []
        
        DataBase = DB()
        DataBase.connect()
        sql_sig_getSegments ='select ppg,label_arrhythmia,label_artifacts,param_id,id from segments'

        Segments_rows = DataBase.select(sql_sig_getSegments)
        for Segments_row in Segments_rows:
            
        #    if Segments_row[2] == 0: #只留下沒有noise的資料

                arrhythmia_Label = Segments_row[1]
                # if(Segments_row[1] > 0): #改成分兩類
                #     arrhythmia_Label = 1
                
                if (Segments_row[3]%3)!=0:

                    self.datasTrain.append(blob_to_array(Segments_row[0]))
                    self.labelsTrain.append(arrhythmia_Label)
                    self.dataIdTrain.append(Segments_row[4])

                else:

                    self.datasTest.append(blob_to_array(Segments_row[0]))
                    self.labelsTest.append(arrhythmia_Label)
                    self.dataIdTest.append(Segments_row[4])



        DataBase.close()


    def __len__(self):
        """'return the size of dataset"""

        if self.mode == 'train':
            return len(self.datasTrain)
        else:
            return len(self.datasTest)

    def __getitem__(self, index):

        if self.mode == 'train':
            single_data = self.datasTrain[index]
            single_label = self.labelsTrain[index]
            single_id = self.dataIdTrain[index]

        else:
            single_data = self.datasTest[index]
            single_label = self.labelsTest[index]
            single_id = self.dataIdTest[index]
        
        single_data = get_features(single_data)
        
        single_data= np.array([single_data])
        #single_data=np.transpose(single_data,(0,2,1))
        
        return (single_data,single_label,single_id)
