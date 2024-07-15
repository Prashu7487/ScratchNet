from ScratchNet.layer import Layer
import numpy as np


def normalize_array(A):
    arr_min = A.min()
    arr_max = A.max()
    if arr_min == arr_max:
        return np.ones_like(A)
    normalized_arr = (A - arr_min) / (arr_max - arr_min)
    return normalized_arr


class Normalize(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        self.input = input
        return normalize_array(input)

    def backward(self, output_gradient, learning_rate):
        return output_gradient
