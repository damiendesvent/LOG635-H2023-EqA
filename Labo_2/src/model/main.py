import numpy as np
import matplotlib.pyplot as plt
from ..preprocessing.split_dataset import split_dataset
from neural_network import NeuralNetwork

Xtrain, Xtest, ytrain, ytest = split_dataset('C:/Users/aq80980/Desktop/prog/LOG635-H2023-EqA/Labo_2/output/clean')


# print(f"Shape of train set is {Xtrain.shape}")
# print(f"Shape of test set is {Xtest.shape}")
# print(f"Shape of train label is {ytrain.shape}")
# print(f"Shape of test labels is {ytest.shape}")


# nn = NeuralNetwork(layers=[40*40,10,8], learning_rate=0.01, iterations=500)
# nn.fit(Xtrain, ytrain)
# nn.plot_loss()