from three_layer_neural_network import NeuralNetwork
import numpy as np
from sklearn import datasets, linear_model
import three_layer_neural_network as nn
import matplotlib.pyplot as plt

class DeepNeuralNetwork(NeuralNetwork):
    def __init__(self, hidden_nodes, actFun_type, reg_lambda=0.01, seed=0):
        '''
        Parameters:
        --------
        hidden_nodes: list of hidden nodes in each layer (including input layer), type: list of int
                     requires hidden_nodes[0] == dim_input hidden_nodes[-1] == num_classes
        actFun_type: list of activation functions corresponding to each layer, type: list of str with same lens of hidden_nodes
        reg_lambda: regularization coefficient, type: float
        seed: random seed, type: int
        '''

        self.hidden_nodes = hidden_nodes
        self.reg_lambda = reg_lambda

        np.random.seed(seed)

        self.num_layer = len(self.hidden_nodes)
        # sanity check
        assert len(actFun_type) == self.num_layer - 2

        for i in range(len(actFun_type)):
            assert actFun_type[i] in ('Tanh', 'Sigmoid', 'ReLU')

        # initialize the weights and biases in the network
        self.W = {}
        self.b = {}
        self.actFun_type = {}

        for i in range(1, self.num_layer):
            self.W[i] = np.random.randn(self.hidden_nodes[i - 1], self.hidden_nodes[i]) / np.sqrt(
                self.hidden_nodes[i - 1])
            self.b[i] = np.zeros((1, self.hidden_nodes[i]))
            if i != (self.num_layer - 1):
                self.actFun_type[i] = actFun_type[i - 1]
    def feedforward(self, X):
        self.z = {}
        self.a = {0: X}

        # Inner layers
        for i in range(1, self.num_layer - 1):
            self.z[i] = self.a[i - 1] @ self.W[i] + self.b[i]
            self.a[i] = self.actFun(self.z[i], type = self.actFun_type[i])
        # Output layer
        i = self.num_layer - 1
        self.z[i] = self.a[i - 1] @ self.W[i] + self.b[i]
        self.a[i] = np.exp(self.z[i]) / np.sum(np.exp(self.z[i]), axis = 1, keepdims = True)
        return None


    def calculate_loss(self, X, y):
        '''
        calculate the loss function

        Parameters:
        --------
        X: input data, type: numpy ndarray with shape [num_data, self.hidden_nodes[0]]
        y: input label, type: numpy array with shape [num_data, 1]

        Return:
        ------
        loss: calculated loss, type: float
        '''
        num_examples = len(X)
        self.feedforward(X)

        data_loss = - np.sum(np.log(self.a[self.num_layer - 1]) * np.eye(2)[y])
        # Add regularization term to loss
        data_loss += self.reg_lambda / self.num_layer * np.sum([np.sum(np.square(i)) for i in self.W])
        return (1. / num_examples) * data_loss
    def backprop(self, X, y):
        # define dW and db so that they have same len as W and b
        dW = {}
        db = {}
        delta = {}

        num_examples = len(X)
        delta[self.num_layer] = self.a[self.num_layer - 1]
        # y_ - y
        delta[self.num_layer][range(num_examples), y] -= 1

        for i in range(1, self.num_layer)[::-1]:
            dW[i] = self.a[i - 1].T @ delta[i + 1]
            db[i] = np.mean(delta[i + 1], axis=0)
            if i != 1:
                delta[i] = delta[i + 1] @ self.W[i].T * self.diff_actFun(self.z[i - 1], type=self.actFun_type[i - 1])
        return dW, db

    def predict(self, X):
        '''
        predict the labels of a given data point X
        Parameters:
        ------
        X: input data, type: numpy ndarray with shape [num_data, self.hidden_nodes[0]]
        Return:
        ------
        estimated label of X
        '''
        self.feedforward(X)
        return np.argmax(self.a[self.num_layer - 1], axis=1)

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        '''
        uses backpropagation and GD (gradient descent) to train the model

        Parameters:
        --------
        X: input data, type: numpy ndarray with shape [num_data, self.hidden_nodes[0]]
        y: input label, type: numpy array with shape [num_data, 1]
        epsilon: step size for each parameter, type: float
        num_passes: number of iterations to train the model, type: int
        print_loss: steps taken to print loss, type: int
        '''

        for i in range(0, num_passes):
            # feedforward propagation
            self.feedforward(X)
            # backpropagation
            dW, db = self.backprop(X, y)

            for j in range(1, self.num_layer):
                # add regularization to dW
                dW[j] += self.reg_lambda * self.W[j]  # add regularization to dW

                # GD parameter update
                self.W[j] += -epsilon * dW[j]
                self.b[j] += -epsilon * db[j]

            # Optionally print out the loss
            if print_loss and i % 1000 == 0:
                print('Loss after iteration %i: %f' % (i, self.calculate_loss(X, y)))

class Layer(object):
# implements the feedforward and backprop steps for a single layer in the network
    def actFun(self, z, type):
        '''
        actFun computes the activation functions
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: activations
        '''

        if type == 'Tanh':
            return np.tanh(z)
        elif type == 'Sigmoid':
            return 1. / (1. + np.exp(-z))
        elif type == 'ReLU':
            return z * (z > 0)
        else:
            return None
    def ff(self, z_in, W, b, actFuntype):
        '''
        feedforward builds a 3-layer neural network and computes the two probabilities,
        one for class 0 and one for class 1
        :param X: input data
        :param actFun: activation function
        :return:
        '''

        z1 = z_in @ W + b
        a1 = self.actFun(z1, type = actFuntype)

        return a1, z1

    def backprop(self, delta_in, a):
        '''
        backprop implements backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param y: given labels
        :return: dL/dW1, dL/b1, dL/dW2, dL/db2
        '''

        dW = a.T @ delta_in
        db = np.mean(delta_in, axis=0)
        return dW, db

def main():

    # generate and visualize Make-Moons dataset
    X, y = nn.generate_data()
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.show()

    # neunral network configuration
    hidden_nodes = [2, 10, 10, 10, 10, 2]
    actFun_type = ['Sigmoid'] * (len(hidden_nodes) - 2)

    # model training and result showing
    model = DeepNeuralNetwork(hidden_nodes = hidden_nodes, actFun_type = actFun_type)
    model.fit_model(X, y)
    model.visualize_decision_boundary(X, y)

if __name__ == "__main__":
    main()