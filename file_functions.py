# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 16:44:58 2019
@author: dorgans
"""

import numpy as np 
import pandas as pd
import igor.binarywave as bw
import datetime
import easygui
import os
import pickle

from matplotlib import cm
from matplotlib import pyplot as plt
from neo import io


def load_directory_content__(): #   This lists waves from selected directory
    directory = easygui.diropenbox(default=r'Z:\')
    paths__ = []
    for i in range(len(os.listdir(directory))):
        paths__.append( directory+"\\"+os.listdir(directory)[i])
    return (paths__, directory)

def load_directory_content_and_sub__(): #   This lists waves from selected directory
    directory = easygui.diropenbox(default=r'Z:\')
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(directory):
        for file in f:
            files.append(os.path.join(r, file))
    return (files , directory)

def load_specific_file__(default__=r'Z:\UusisaariU\Kevin\ePhy'): #   This lists waves from selected directory
    path__ = easygui.fileopenbox(default=default__)
    return (path__)

def load_IgorWave__(path): #    This reads Igor wave
    r = io.IgorIO(filename=path)
    data = bw.load(path)
    timestamp = int(data["wave"]["wave_header"]["modDate"])
    analogsignal_number = int(data["wave"]['wave_header']['nDim'][1])
    sampling__ = float(r.read_analogsignal().sampling_rate)
    try:
        sweep_position__ = int(r.filename.split('\\R')[1].split('_')[0])
    except:
        sweep_position__ = np.nan
    if analogsignal_number>1:
        sweep_lag = np.array(str(data["wave"]['note']).split('%SweepStartTimes^>')[1].split('|')[0:analogsignal_number], dtype=np.float64)
    else:
        sweep_lag = [0]
    return r, sampling__, sweep_position__, timestamp, sweep_lag

    
def save_analysis__(directory__, kwargs__, list__): #This creates CSV matrix with extracted parameters
    #   1- if file does not exist > creates one
    #   2- if file exists but element not in list > add line with new element
    #   3- if file exists and element is in list > overwrites it
    try:
        file__ = np.transpose(np.array(pd.read_csv(r"Z:\UusisaariU\Kevin\ePhy\GluUcG\AnalysisMatrix["+ directory__.split('\\')[len(directory__.split('\\'))-2] +"].csv", header=None)))
        file__ = file__[1::]
        if (list__[0] in np.transpose(file__)[0])==True: #if element already in list, overwrite on line
            index__ = np.transpose(file__)[0].tolist().index(list__[0])
            for i in range(len(list__)):
                file__[index__][i] = list__[i]
            file__ = np.transpose(file__)
        else:   #Write new element in list if it doesn't exist
            file__ = np.concatenate((file__, [list__]))
            file__ = np.transpose(file__)

    except:
        print('/// creating AnalysisMatrix.csv ///')
        file__ = np.transpose(list__)
    pd.DataFrame(file__, index=kwargs__).to_csv(r"Z:\UusisaariU\Kevin\ePhy\GluUcG\AnalysisMatrix["+ directory__.split('\\')[len(directory__.split('\\'))-2] +"].csv", index=True, index_label=kwargs__, header=None)

#CREATE ASOCIATED DATABASE WITH PICKLE
def save_pickle_db(array_content):
    PATH = load_directory_content__()
    DATABASE_NAME = r'\NEWdb.pyobj'
    with open(PATH[1]+DATABASE_NAME, 'wb') as f: 
        pickle.dump(array_content, f)

#THIS WILL LOAD PICKLE DATABASE
def load_pickle_db():
    PATH = load_specific_file__()
    with open(PATH, 'rb') as f:  # Python 3: open(..., 'rb')
        array_content = pickle.load(f)
    return (array_content)

def load_pickle_file(file):
    with open(file, 'rb') as f:  # Python 3: open(..., 'rb')
        array_content = pickle.load(f)
    return (array_content)


def easy_IbwLoad(DOWNSAMPLING__=1, _file_ = ''):
    if _file_ == '':
        r, sampling__, sweep_position__, start_time, sweep_lag = load_IgorWave__(load_specific_file__())
    else:
        r, sampling__, sweep_position__, start_time, sweep_lag = load_IgorWave__(_file_)
    signal = r.read_analogsignal().transpose()
    print(datetime.datetime.fromtimestamp(start_time).isoformat())
    for j in range(len(signal)):
        RAW = np.array(signal[j], dtype=np.float64)*1000
    return RAW, sampling__, sweep_position__, start_time, sweep_lag 

