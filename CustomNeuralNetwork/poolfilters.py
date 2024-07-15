import numpy as np
from CustomNeuralNetwork.layer import Layer
import warnings


class MaxPooling2D(Layer):
    def __init__(self, pool_size=(2, 2), stride=2):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, input):
        self.input = input
        channels, height, width = input.shape
        pool_height, pool_width = self.pool_size
        out_height = (height - pool_height) // self.stride + 1
        out_width = (width - pool_width) // self.stride + 1

        self.output = np.zeros((channels, out_height, out_width))

        for h in range(out_height):
            for w in range(out_width):
                # defining filter space
                h_start = h * self.stride
                h_end = h_start + pool_height
                w_start = w * self.stride
                w_end = w_start + pool_width
                for c in range(channels):
                    self.output[c, h, w] = np.max(input[c, h_start:h_end, w_start:w_end])
        return self.output

    def backward(self, output_gradient, learning_rate):
        channels, height, width = self.input.shape
        pool_height, pool_width = self.pool_size
        out_height = (height - pool_height) // self.stride + 1
        out_width = (width - pool_width) // self.stride + 1

        input_gradient = np.zeros_like(self.input)

        for h in range(out_height):
            for w in range(out_width):
                for c in range(channels):
                    h_start = h * self.stride
                    h_end = h_start + pool_height
                    w_start = w * self.stride
                    w_end = w_start + pool_width

                    window = self.input[c, h_start:h_end, w_start:w_end]
                    max_val = np.max(window)
                    mask = (window == max_val)

                    input_gradient[c, h_start:h_end, w_start:w_end] += mask * output_gradient[c, h, w]

        return input_gradient


# yet to improve c,h,w thing
class AveragePooling2D(Layer):
    def __init__(self, pool_size=(2, 2), stride=2):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, input):
        self.input = input
        height, width, channels = input.shape
        pool_height, pool_width = self.pool_size
        out_height = (height - pool_height) // self.stride + 1
        out_width = (width - pool_width) // self.stride + 1

        self.output = np.zeros((out_height, out_width, channels))

        for h in range(out_height):
            for w in range(out_width):
                for c in range(channels):
                    h_start = h * self.stride
                    h_end = h_start + pool_height
                    w_start = w * self.stride
                    w_end = w_start + pool_width
                    self.output[h, w, c] = np.mean(input[h_start:h_end, w_start:w_end, c])

        return self.output

    def backward(self, output_gradient, learning_rate):
        height, width, channels = self.input.shape
        pool_height, pool_width = self.pool_size
        out_height = (height - pool_height) // self.stride + 1
        out_width = (width - pool_width) // self.stride + 1

        input_gradient = np.zeros_like(self.input)

        for h in range(out_height):
            for w in range(out_width):
                for c in range(channels):
                    h_start = h * self.stride
                    h_end = h_start + pool_height
                    w_start = w * self.stride
                    w_end = w_start + pool_width

                    gradient = output_gradient[h, w, c] / (pool_height * pool_width)
                    input_gradient[h_start:h_end, w_start:w_end, c] += gradient

        return input_gradient
