import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

class NeuralNetwork():
    def __init__(self, nb_input_nodes, nb_hidden_nodes, nb_output_nodes, learning_rate):
        #  >>> HYPERPARAMETERS <<<
        self.nb_input_nodes = nb_input_nodes
        self.nb_output_nodes = nb_output_nodes
        self.nb_hidden_nodes = nb_hidden_nodes
        self.learning_rate = learning_rate
        self.losses = []

        # >>> WEIGHTS + BIASES <<<
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

    # def derivative_sigmoid(self, z):
    #     return self.sigmoid(z) * (1 - self.sigmoid(z))

    def entropy_loss(self, y, y_pred):
        eps = np.finfo(float).eps = 0
        loss = -np.sum(y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps))
        return loss
    
    # def derivative_entropy_loss(self, y, y_pred):
    #     eps = np.finfo(float).eps
    #     return np.divide((1 - y), (1 - y_pred + eps)) - np.divide(y, y_pred + eps)

    def forward(self, X):
        # Forward propagation through our network
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

        # dCost_dA3 = self.derivative_entropy_loss(y, self.A3)
        # dA3_dZ3 = self.A3 * (1 - self.A3) # derivative_sigmoid = s(z) * (1 - s(z)) and s(z_n) = a_n
        # dCost_dZ3 = self.A3 - y # = dCost_dA3 * dA3_dZ3
        # dZ3_dW2 = self.A2.T # (20, 1) to outer product with dCost_dZ3 (1, 8)
        # dZ3_dA2 = self.W2.T
        # dCost_dW2 = np.dot(dCost_dZ3, dZ3_dW2.T) # self.A3.T 
        # dCost_dA2 = np.dot(dCost_dZ3, dZ3_dA2)
        # dCost_db2 = dCost_dZ3 # dZ3_db2 is identity

        # dA2_dZ2 = self.A2 * (1 - self.A2) # derivative_sigmoid = s(z) * (1 - s(z)) and s(z_n) = a_n
        # dCost_dZ2 = np.multiply(dCost_dA2, dA2_dZ2)
        # dZ2_dW1 = self.X.T # (1600, 1) to outer product with dCost_dZ3 (1, 20)
        # dCost_dW1 = np.dot(dZ2_dW1.T, dCost_dZ2)
        # dCost_db1 = dCost_dZ2 # dZ2_db1 is identity


        # dCost_db2 = np.dot(np.ones(N), (self.A2 - y))
        # dCost_dW2 = np.dot(self.A1.T, (self.A2 - y))
        # delta1 = np.dot((self.A2 - y), self.W2.T)
        # dCost_db1 =  np.dot(np.ones(N), delta1 * self.A1 * (1 - self.A1))
        # dCost_dW1 = np.dot(self.X.T, delta1 * self.A1 * (1 - self.A1))

        N = y.shape[0]
        dL_dZ2 = (self.A2 - y)
        dL_dW2 = self.A1.T @ dL_dZ2
        ones = np.ones((N, 1))
        dL_db2 = ones.T @ dL_dZ2
        delta1 = dL_dZ2 @ self.W2.T
        dsig = self.sigmoid(self.A1) * (1 - self.sigmoid(self.A1))
        dL_dW1 = self.X.T @ (delta1 * dsig)
        dL_db1 = ones.T @ (delta1 * dsig)

        # Wights and biases update
        self.W2 = self.W2 - self.learning_rate * dL_dW2
        self.b2 = self.b2 - self.learning_rate * dL_db2
        self.W1 = self.W1 - self.learning_rate * dL_dW1
        self.b1 = self.b1 - self.learning_rate * dL_db1

    def train(self, X, y, epochs):
        for i in range(epochs):
            y_pred = self.forward(X)
            loss = self.entropy_loss(y, y_pred)
            self.losses.append(loss)
            self.backward(y)

            # if i == 0 or i == nb_iterations-1:
            #     print(f"Iteration: {i+1}")
            #     print(tabulate(zip(X, y, [np.round(y_pred) for y_pred in self.A3] ), headers=["Input", "Actual", "Predicted"]))
            #     print(f"Loss: {loss}")                
            #     print("\n")

    def predict(self, X):
        return np.round(self.forward(X))

    def plot_loss(self):
        plt.plot(self.losses)
        plt.xlabel("Iteration")
        plt.ylabel("loss")
        plt.title("Loss curve for training")
        plt.show()