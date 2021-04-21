import json
import numpy as np
import os
import random
import scipy.io as sio
import tqdm
from scipy import signal
from scipy.signal import butter, iirnotch

def butterworth_filter(data, fs=300, order=5, cutoff_high=1, cutoff_low=40, powerline=60):
    b, a = butter(order, cutoff_high/(0.5*fs), btype='high', analog=False, output='ba')
    x = signal.filtfilt(b, a, data)
    d, c = butter(order, cutoff_low/(0.5*fs), btype='low', analog=False, output='ba')
    y = signal.filtfilt(d, c, x)
    f, e = iirnotch(powerline/(0.5*fs), 30)
    z = signal.filtfilt(f, e, y)     
    return z

STEP = 256

def load_ecg_mat(ecg_file):
    return sio.loadmat(ecg_file)['val'].squeeze()

def load_all(data_path):
    label_file = os.path.join(data_path, "REFERENCE.csv")
    with open(label_file, 'r') as fid:
        records = [l.strip().split(",") for l in fid]

    dataset = []
    for record, label in tqdm.tqdm(records):
        if label == "~" :
            pass
        else:
            ecg_file = os.path.join(data_path, record + ".mat")
            ecg_file = os.path.abspath(ecg_file)
            ecg = load_ecg_mat(ecg_file)
            ecg = butterworth_filter(ecg)
            num_labels = ecg.shape[0] / STEP
            dataset.append((ecg_file, [label]*num_labels))
    return dataset 

def make_json(save_path, dataset):
    with open(save_path, 'w') as fid:
        for d in dataset:
            datum = {'ecg' : d[0],
                     'labels' : d[1]}
            json.dump(datum, fid)
            fid.write('\n')

def make_txt(save_path):
    label_file = os.path.join(data_path, "REFERENCE.csv")
    with open(save_path, "w") as ansfile:
        with open(label_file, 'r') as fid:
            for l in fid:
                [name, ans] = l.strip().split(",") 
                if ans == "~":
                    pass
                else:
                    print >> ansfile, name+", "+ans


if __name__ == "__main__":

    data_path = "data/sample2017/validation/"
    sample_dataset = load_all(data_path)
    make_json("data/sample2017/sample.json", sample_dataset)
    make_txt("data/sample2017/ansfile.txt")

#/content/ecg/examples/cinc17/data/sample2017/validation/REFERENCE.csv
