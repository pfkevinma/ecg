import json
import numpy as np
import os
import random
import scipy.io as sio
import tqdm

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

if __name__ == "__main__":

    data_path = "data/sample2017/validation/"
    sample_dataset = load_all(data_path)
    make_json("data/sample2017/sample.json", sample_dataset)

#/content/ecg/examples/cinc17/data/sample2017/validation/REFERENCE.csv
