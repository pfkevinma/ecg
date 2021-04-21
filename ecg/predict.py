from __future__ import print_function

import argparse
import numpy as np
import keras
import os
import scipy.stats as sst

import load
import util


def predict(data_json, model_path):
    preproc = util.load(os.path.dirname(model_path))
    dataset = load.load_dataset(data_json)
    x, y = preproc.process(*dataset)

    model = keras.models.load_model(model_path)
    probs = model.predict(x, verbose=1)

    return probs, preproc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_json", help="path to data json")
    parser.add_argument("model_path", help="path to model")
    args = parser.parse_args()
    probs, preproc = predict(args.data_json, args.model_path)

    ans = sst.mode(np.argmax(probs, axis=2), axis=1)[0]

    file = open("predict_answers.txt", "w")
    for row in ans:
        file.write(preproc.classes[row[0]])
        file.write("\n")
    file.close()
