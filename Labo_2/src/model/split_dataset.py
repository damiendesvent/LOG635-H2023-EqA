import cv2
from files_list import files_list
import numpy as np

from os.path import basename, dirname

percentage_test = .2

def one_hot_encode(category, categories):
    category_index = categories.index(category)
    y_vector = np.zeros(8)
    y_vector[category_index] = 1
    return y_vector

def modulo_shuffle(list, step):
    length = len(list)
    while length % step == 0:
        step = step + 1
    i = 0
    while i != length - step:
        yield list[i]
        i = (i + step) % length

def count_each_shape(files, categories):
    names = list(map(lambda f: basename(dirname(f)), files))
    return list(map(lambda category: (category, names.count(category)), categories))

def split_dataset(data_root, categories):
    Xtrain = []
    Xtest = []
    ytrain = []
    ytest = []
    files = files_list(data_root)
    print(count_each_shape(files, categories))
    test_set_size = percentage_test * len(files)
    # access files out of order to prevent having all the same shapes in the test set
    for filepath in modulo_shuffle(files, 305):
        label = basename(dirname(filepath))
        y_vector = one_hot_encode(label, categories)
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
