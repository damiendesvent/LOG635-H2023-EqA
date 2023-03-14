import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork():
    def __init__(self, nb_input_nodes, nb_hidden_nodes, nb_output_nodes, learning_rate, epochs):
        self.nb_input_nodes = nb_input_nodes
        self.nb_output_nodes = nb_output_nodes
        self.nb_hidden_nodes = nb_hidden_nodes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.losses_train = []
        self.losses_valid = []

        # Weight matrix from input to hidden layer
        self.W1 = np.random.randn(self.nb_input_nodes, self.nb_hidden_nodes)
        # Bias from input to hidden layer
        self.b1 = np.random.randn(1, self.nb_hidden_nodes)

        # Weight matrix from hidden to output layer
        self.W2 = np.random.randn(self.nb_hidden_nodes, self.nb_output_nodes)
        # Bias from hidden to output layer
        self.b2 = np.random.randn(1, self.nb_output_nodes)

    def sigmoid(self, z):
        s = 1 / (1 + np.exp(-z))
        return s

    def entropy_loss(self, y, y_pred):
        N = y.shape[0]
        eps = np.finfo(float).eps
        y_pred = np.maximum(y_pred, eps)
        y_pred = np.minimum(y_pred, 1 - eps)
        loss = -np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)) / (N * self.nb_output_nodes)
        return loss
    
    def forward(self, X):
        self.X = X
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.sigmoid(self.Z2)
        return self.A2

    def backward(self, y):
        
        # https://www.youtube.com/watch?v=tIeHLnjs5U8 Backpropagation calculus
        # https://www.youtube.com/watch?v=x4RNPJD-IkQ Backpropagation: Compute the Derivatives - Part 1
        # https://www.youtube.com/watch?v=JsbFBJCWbeI Backpropagation: Compute the Derivatives - Part 2
        # https://www.youtube.com/watch?v=55nIWdjgOJU Code a NN from Scratch

        N = y.shape[0]
        dL_dZ2 = (self.A2 - y)
        dL_dW2 = self.A1.T @ dL_dZ2
        ones = np.ones((N, 1))
        dL_db2 = ones.T @ dL_dZ2
        delta1 = dL_dZ2 @ self.W2.T
        dsig = self.sigmoid(self.A1) * (1 - self.sigmoid(self.A1))
        dL_dW1 = self.X.T @ (delta1 * dsig)
        dL_db1 = ones.T @ (delta1 * dsig)

        # Update weights and biases
        self.W2 = self.W2 - self.learning_rate * dL_dW2
        self.b2 = self.b2 - self.learning_rate * dL_db2
        self.W1 = self.W1 - self.learning_rate * dL_dW1
        self.b1 = self.b1 - self.learning_rate * dL_db1

    def train(self, Xtrain, ytrain, Xvalid, yvalid):
        for i in range(self.epochs):
            # valid
            yvalid_pred = self.forward(Xvalid)
            loss_valid = self.entropy_loss(yvalid, yvalid_pred)
            self.losses_valid.append(loss_valid)
            # train
            ytrain_pred = self.forward(Xtrain)
            loss_train = self.entropy_loss(ytrain, ytrain_pred)
            self.losses_train.append(loss_train)
            self.backward(ytrain)


            # if i == 0 or i == nb_iterations-1:
            #     print(f"Iteration: {i+1}")
            #     print(tabulate(zip(X, y, [np.round(y_pred) for y_pred in self.A3] ), headers=["Input", "Actual", "Predicted"]))
            #     print(f"Loss: {loss}")                
            #     print("\n")

    def predict(self, X):
        return np.round(self.forward(X))

    def plot_loss(self, name):
        plt.plot(self.losses_train)
        plt.plot(self.losses_valid)
        plt.xlabel("Iteration")
        plt.ylabel("loss")
        plt.title("Loss curve for training")
        plt.savefig(name)
        plt.clf()
        # plt.show()
