import numpy as np
from CustomNeuralNetwork.layer import Layer
import warnings


class Dense(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.randn(output_size, input_size)   #instead of transpose weight is even created that way
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = input
        self.output = np.dot(self.weights, self.input) + self.bias

        # if np.isnan(self.output).any():
        #     print("input shape", self.input.shape)
        #     print("weights", self.weights)
        #     print("biases", self.bias)
        #     print("input", self.input)
        #     print("op", self.output)
        #     warnings.warn("dense output has NaN")
        return self.output

    def backward(self, output_gradient, learning_rate):
        # print("output_gradient in dense backpass: ",output_gradient)
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient
