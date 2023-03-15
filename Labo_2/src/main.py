from model.split_dataset import split_dataset
from model.neural_network import NeuralNetwork
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score
import os

data_root = str(Path(__file__, '../../output/clean').resolve())
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

[[Xtrain, ytrain], [Xvalid, yvalid], [Xtest, ytest]] = split_dataset(data_root, categories, [8, 1, 1])
print(f"Shape of Xtrain is {Xtrain.shape}")
print(f"Shape of ytrain is {ytrain.shape}")
print(f"Shape of Xvalid is {Xvalid.shape}")
print(f"Shape of yvalid is {yvalid.shape}")
print(f"Shape of Xtest is {Xtest.shape}")
print(f"Shape of ytest is {ytest.shape}")


for i in range(70, 10, -10):

  nn = NeuralNetwork(
    nb_input_nodes=40*40,
    nb_hidden_nodes=i,
    nb_output_nodes=8,
    learning_rate=0.001,
    epochs=100
  )

  nn.train(Xtrain, ytrain, Xvalid, yvalid)

  yvalid_pred = nn.predict(Xvalid)
  accuracy = accuracy_score(yvalid.argmax(axis=1), yvalid_pred.argmax(axis=1))
  
  # ytest_pred = nn.predict(Xtest)
  # accuracy = accuracy_score(ytest.argmax(axis=1), ytest_pred.argmax(axis=1))

  # print('train')
  # ytrain_pred = nn.predict(Xtrain)
  # print(confusion_matrix(ytrain.argmax(axis=1), ytrain_pred.argmax(axis=1)))
  # print(accuracy_score(ytrain.argmax(axis=1), ytrain_pred.argmax(axis=1)))
  # print('test')
  # ytest_pred = nn.predict(Xtest)
  # print(confusion_matrix(ytest.argmax(axis=1), ytest_pred.argmax(axis=1)))
  # print(accuracy_score(ytest.argmax(axis=1), ytest_pred.argmax(axis=1)))
  # print(f"Final loss train: {nn.losses_train[-1]}")
  # print(f"Final loss test: {nn.losses_test[-1]}")

  # wrong_classifications_train = [(image, categories[y], categories[predicted]) for (image, y, predicted) in zip(Xtrain, ytrain.argmax(axis=1), ytrain_pred.argmax(axis=1)) if y != predicted ]
  # wrong_classifications_test = [(image, categories[y], categories[predicted]) for (image, y, predicted) in zip(Xtest, ytest.argmax(axis=1), ytest_pred.argmax(axis=1)) if y != predicted ]
  # print(f"wrong_classifications_train: {len(wrong_classifications_train)}")
  # print(f"wrong_classifications_test: {len(wrong_classifications_test)}")
  # print(wrong_classifications_train)
  # print(wrong_classifications_test)

  trainingInfo_path = r"Labo_2/trainingInfo/"
  idx = len([entry for entry in os.listdir(trainingInfo_path) if os.path.isfile(os.path.join(trainingInfo_path, entry))])


  nn.plot_loss(f"{trainingInfo_path}Loss_accuracy{accuracy}_hidden{nn.nb_hidden_nodes}_rate{nn.learning_rate}_epochs{nn.epochs}datasize{Xtrain.shape[0]}_{idx}.pdf")