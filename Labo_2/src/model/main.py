import numpy as np
import matplotlib.pyplot as plt
from split_dataset import split_dataset
from neural_network import NeuralNetwork
from pathlib import Path


data_root = str(Path(__file__, '../../../output/clean').resolve())
print(data_root)

categories = [
  'Cercle2',
  'Cercle5',
  'Diamant2',
  'Diamant5',
  'Hexagone2',
  'Hexagone5',
  'Triangle2',
  'Triangle5',
]

Xtrain, Xtest, ytrain, ytest = split_dataset(data_root, categories)
print(f"Shape of train set is {Xtrain.shape}")
print(f"Shape of test set is {Xtest.shape}")
print(f"Shape of train label is {ytrain.shape}")
print(f"Shape of test labels is {ytest.shape}")


nn = NeuralNetwork(layers=[40*40,10,8], learning_rate=0.01, iterations=500)
nn.fit(Xtrain, ytrain)
nn.plot_loss()