from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import json
import keras
import numpy as np
import os
import random
import scipy.io as sio
import tqdm
from scipy import signal
from scipy.signal import butter, iirnotch

STEP = 256

def data_generator(batch_size, preproc, x, y):
    num_examples = len(x)
    examples = zip(x, y)
    examples = sorted(examples, key = lambda x: x[0].shape[0])
    end = num_examples - batch_size + 1
    batches = [examples[i:i+batch_size]
                for i in range(0, end, batch_size)]
    random.shuffle(batches)
    while True:
        for batch in batches:
            x, y = zip(*batch)
            yield preproc.process(x, y)

class Preproc:

    def __init__(self, ecg, labels):
        self.mean, self.std = compute_mean_std(ecg)
        self.classes = sorted(set(l for label in labels for l in label))
        self.int_to_class = dict( zip(range(len(self.classes)), self.classes))
        self.class_to_int = {c : i for i, c in self.int_to_class.items()}

    def process(self, x, y):
        return self.process_x(x), self.process_y(y)

    def process_x(self, x):
        x = pad(x)
        x = (x - self.mean) / self.std
        x = x[:, :, None]
        return x

    def process_y(self, y):
        # TODO, awni, fix hack pad with noise for cinc
        y = pad([[self.class_to_int[c] for c in s] for s in y], val=2, dtype=np.int32) 
        y = keras.utils.np_utils.to_categorical(
                y, num_classes=len(self.classes))
        return y

def pad(x, val=0, dtype=np.float32):
    max_len = max(len(i) for i in x)
    padded = np.full((len(x), max_len), val, dtype=dtype)
    for e, i in enumerate(x):
        padded[e, :len(i)] = i
    return padded

def compute_mean_std(x):
    x = np.hstack(x)
    return (np.mean(x).astype(np.float32),
           np.std(x).astype(np.float32))

def load_dataset(data_json):
    with open(data_json, 'r') as fid:
        data = [json.loads(l) for l in fid]
    labels = []; ecgs = []
    for d in tqdm.tqdm(data):
        labels.append(d['labels'])
        ecgs.append(load_ecg(d['ecg']))
    return ecgs, labels

def load_ecg(record):
    if os.path.splitext(record)[1] == ".npy":
        ecg = np.load(record)
    elif os.path.splitext(record)[1] == ".mat":
        ecg = sio.loadmat(record)['val'].squeeze()
        ecg = butterworth_filter(ecg)
    else: # Assumes binary 16 bit integers
        with open(record, 'r') as fid:
            ecg = np.fromfile(fid, dtype=np.int16)

    trunc_samp = STEP * int(len(ecg) / STEP)
    return ecg[:trunc_samp]

def butterworth_filter(data, fs=300, order=5, cutoff_high=1, cutoff_low=40, powerline=60):
    b, a = butter(order, cutoff_high/(0.5*fs), btype='high', analog=False, output='ba')
    x = signal.filtfilt(b, a, data)
    d, c = butter(order, cutoff_low/(0.5*fs), btype='low', analog=False, output='ba')
    y = signal.filtfilt(d, c, x)
    f, e = iirnotch(powerline/(0.5*fs), 30)
    z = signal.filtfilt(f, e, y)     
    return z

if __name__ == "__main__":
    data_json = "cinc17/train.json"
    train = load_dataset(data_json)
    preproc = Preproc(*train)
    gen = data_generator(32, preproc, *train)
    for x, y in gen:
        print(x.shape, y.shape)
        break
    # x, y = preproc.process(*train)
    # print(x.shape, y.shape)
    # with open('inspect_data_x', 'w') as f:
    #     for ele in x[0]:
    #         f.write(str(ele)+'\n')
    #
    # with open('inspect_data_y', 'w') as f:
    #     f.write(str(y[0])+'\n')

