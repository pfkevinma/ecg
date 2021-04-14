from __future__ import print_function

import argparse
import numpy as np
import keras
import os

import load
import util

def predict(data_json, model_path):
    preproc = util.load(os.path.dirname(model_path))
    dataset = load.load_dataset(data_json)
    x, y = preproc.process(*dataset)

    model = keras.models.load_model(model_path)
    probs = model.predict(x, verbose=1)

    return probs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_json", help="path to data json")
    parser.add_argument("model_path", help="path to model")
    args = parser.parse_args()
    probs = predict(args.data_json, args.model_path)
    
    file = open("test.txt", "w")
    for row in probs:
        np.savetxt(file, row)
    file.close()
    
    # prediction = [sst.mode(np.argmax(prob, axis=2).squeeze())[0][0] for prob in probs]
    # return preproc.int_to_class[prediction]

    # for prob in probs:
    #     prediction = sst.mode(np.argmax(prob, axis=2).squeeze())[0][0]
    #     print(preproc.int_to_class[prediction])

