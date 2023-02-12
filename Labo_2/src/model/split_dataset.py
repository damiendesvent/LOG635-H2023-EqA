import cv2
from files_list import files_list
import numpy as np

from os.path import basename, dirname

percentage_test = .2

def split_dataset(data_root, categories):
    Xtrain = []
    Xtest = []
    ytrain = []
    ytest = []
    files = files_list(data_root)
    test_set_size = percentage_test * len(files)
    for filepath in files:
        label = basename(dirname(filepath))
        category = categories.index(label)
        y_vector = np.zeros(8)
        y_vector[category] = 1
        image = cv2.cvtColor(cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2GRAY)
        if len(ytest) < test_set_size:
            Xtest.append(image)
            ytest.append(y_vector)
        else:
            Xtrain.append(image)
            ytrain.append(y_vector)
    Xtrain = np.array(Xtrain, dtype=np.int32)
    Xtest = np.array(Xtest, dtype=np.int32)
    ytrain = np.array(ytrain)
    ytest = np.array(ytest)
    return (
        Xtrain.reshape(Xtrain.shape[0], -1),
        Xtest.reshape(Xtest.shape[0], -1),
        ytrain,
        ytest
    )