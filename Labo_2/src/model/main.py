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
print(f"Shape of Xtrain is {Xtrain.shape}")
print(f"Shape of Xtest is {Xtest.shape}")
print(f"Shape of ytrain is {ytrain.shape}")
print(f"Shape of ytest is {ytest.shape}")


nn = NeuralNetwork(
  nb_input_nodes=40*40,
  nb_hidden_nodes=1500,
  nb_output_nodes=8,
  learning_rate=0.000001
)
nn.train(Xtrain, ytrain, epochs=10)
correct_classification = 0
wrong_classification = 0
zero_classification = 0
# for X, y in zip(Xtest[:10], ytest[:10]):
#   X = np.array([X])
#   print(nn.predict(X)[0], y)
#   if np.array_equal(nn.predict(X)[0], y):
#     correct_classification += 1
#   elif np.array_equal(nn.predict(X)[0], [0, 0, 0, 0, 0, 0, 0, 0]):
#     zero_classification += 1
#   # elif np.array_equal(nn.predict(X), [0, 0, 0, 0, 0, 0, 0, 0]):
#   #   zero_classification += 1
#   else :
#     wrong_classification += 1
nn.plot_loss()
# print(correct_classification, zero_classification, wrong_classification)
print(f"Final loss: {nn.losses[-1]}")
# print(nn.predict(Xtrain[5])[0], ytrain[5])