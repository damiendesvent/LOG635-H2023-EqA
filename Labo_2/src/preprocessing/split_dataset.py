import cv2
from Labo_2.src.util.files_list import files_list

percentage_test = .2

def split_dataset(data_root):
    Xtrain = []
    Xtest = []
    ytrain = []
    ytest = []
    files = files_list(data_root)
    test_set_size = percentage_test * len(files)
    for filepath in files:
        label = filepath.split('/')
        # if len(ytest) < test_set_size:
        #     Xtest.append(cv2.imread(filename))
        #     ytest.append(filename)
        # else:
        #     Xtrain.append(cv2.imread(filename))
        #     ytrain.append(filename)
        print(filepath)
    return (Xtrain, Xtest, ytrain, ytest)