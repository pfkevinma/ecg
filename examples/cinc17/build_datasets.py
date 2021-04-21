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
    label_file = os.path.join(data_path, "../REFERENCE-v3.csv")
    with open(label_file, 'r') as fid:
        records = [l.strip().split(",") for l in fid]

    dataset = []
    for record, label in tqdm.tqdm(records):
        ecg_file = os.path.join(data_path, record + ".mat")
        ecg_file = os.path.abspath(ecg_file)
        ecg = load_ecg_mat(ecg_file)
        num_labels = ecg.shape[0] / STEP
        dataset.append((ecg_file, [label]*num_labels))
    return dataset 

def load_3_classes(data_path):
    label_file = os.path.join(data_path, "../REFERENCE-v3.csv")
    with open(label_file, 'r') as fid:
        records = [l.strip().split(",") for l in fid]

    dataset = []
    for record, label in tqdm.tqdm(records):
        if label == "~":
            pass
        else:
            ecg_file = os.path.join(data_path, record + ".mat")
            ecg_file = os.path.abspath(ecg_file)
            ecg = load_ecg_mat(ecg_file)
            ecg = butterworth_filter(ecg)
            num_labels = ecg.shape[0] / STEP
            dataset.append((ecg_file, [label]*num_labels))
    return dataset 

def split(dataset, dev_frac):
    dev_cut = int(dev_frac * len(dataset))
    random.shuffle(dataset)
    dev = dataset[:dev_cut]
    train = dataset[dev_cut:]
    return train, dev

def make_json(save_path, dataset):
    with open(save_path, 'w') as fid:
        for d in dataset:
            datum = {'ecg' : d[0],
                     'labels' : d[1]}
            json.dump(datum, fid)
            fid.write('\n')

if __name__ == "__main__":
    random.seed(2018)

    dev_frac = 0.1
    data_path = "data/training2017/"
    dataset = load_3_classes(data_path)
    train, dev = split(dataset, dev_frac)
    make_json("train.json", train)
    make_json("dev.json", dev)

#/data/REFERENCE-v3.csv
