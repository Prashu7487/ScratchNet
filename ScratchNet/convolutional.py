import warnings
import numpy as np
from ScratchNet.layer import Layer
from scipy import signal


# def dilate_matrix(matrix, k):
#     if k < 0:
#         warnings.warn("Dilation factor k must be a non-negative integer.")
#         return
#
#     if k == 0:
#         return matrix
#
#     original_rows, original_cols = matrix.shape
#
#     dilated_rows = original_rows + (original_rows - 1) * k
#     dilated_cols = original_cols + (original_cols - 1) * k
#
#     # print("after dilation: ", dilated_rows, dilated_cols)
#     dilated_matrix = np.zeros((dilated_rows, dilated_cols))
#
#     for i in range(original_rows):
#         for j in range(original_cols):
#             dilated_matrix[i * (k + 1)][j * (k + 1)] = matrix[i][j]
#
#     return dilated_matrix


# def add_zeros(matrix, k, axis):
#     """
#         useful when "(input_size-kernel_size)/stride" is not an integer
#     :param matrix:
#     :param k:
#     :param axis
#     :return:
#     """
#
#     if k < 0 or axis not in [0, 1]:
#         warnings.warn("k must be >0 and axis should be 0 or 1")
#         return
#
#     if axis == 0:  # Add rows
#         zeros = np.zeros((k, matrix.shape[1]))
#         new_matrix = np.vstack((matrix, zeros))
#     else:  # Add columns
#         zeros = np.zeros((matrix.shape[0], k))
#         new_matrix = np.hstack((matrix, zeros))
#
#     return new_matrix


# def pad_matrix(matrix, k):
#     if k < 0:
#         warnings.warn("padding length must be >0")
#         return
#
#     original_rows, original_cols = matrix.shape
#
#     padded_rows = original_rows + 2 * k
#     padded_cols = original_cols + 2 * k
#
#     padded_matrix = np.zeros((padded_rows, padded_cols))
#
#     padded_matrix[k:k + original_rows, k:k + original_cols] = matrix
#
#     return padded_matrix


# def convolve2d(input, kernel, stride=1, mode="valid"):
#     kernel_height, kernel_width = kernel.shape[0], kernel.shape[1]
#     input_height, input_width = input.shape[0], input.shape[1]
#
#     if mode == "valid":
#         output_height = (input_height - kernel_height) // stride + 1
#         output_width = (input_width - kernel_width) // stride + 1
#     else:
#         output_height = (input_height + kernel_height - 1) // stride + 1
#         output_width = (input_width + kernel_width - 1) // stride + 1
#         padded_input = np.pad(input, ((kernel_height - 1, kernel_height - 1), (kernel_width - 1, kernel_width - 1)))
#         input = padded_input
#
#     output = np.zeros((output_height, output_width))
#
#     for i in range(0, input_height - kernel_height + 1, stride):
#         for j in range(0, input_width - kernel_width + 1, stride):
#             output[i // stride, j // stride] = np.sum(
#                 input[i:i + kernel_height, j:j + kernel_width] * kernel)
#
#     return output


# def correlate2d(input, kernel, stride=1, mode="valid"):
#     # print("kernel is:", kernel)
#     return convolve2d(input, np.flip(np.flip(kernel, axis=0), axis=1), stride, mode)


# class Convolutional(Layer):
#     def __init__(self, input_shape, kernel_shape, depth, stride=1, mode="valid"):
#         super().__init__()
#         input_depth, input_height, input_width = input_shape
#         kernel_height, kernel_width = kernel_shape
#         self.kernel_shape = kernel_shape
#         self.depth = depth
#         self.input_shape = input_shape
#         self.input_depth = input_depth
#         self.stride = stride
#         self.mode = mode
#
#         if mode == "valid":
#             self.output_height = (input_height - kernel_height) // stride + 1
#             self.output_width = (input_width - kernel_width) // stride + 1
#         elif mode == "full":
#             self.output_height = (input_height + kernel_height - 1) // stride + 1
#             self.output_width = (input_width + kernel_width - 1) // stride + 1
#
#         self.output_shape = (depth, self.output_height, self.output_width)
#         self.kernels_shape = (depth, input_depth, kernel_height, kernel_width)
#         self.kernels = np.random.randn(*self.kernels_shape).astype(np.float32)
#         self.biases = np.random.randn(depth, self.output_height, self.output_width).astype(np.float32)
#
#     def forward(self, input):
#         self.input = input.astype(np.float32)
#         self.output = np.copy(self.biases)
#         for i in range(self.depth):
#             for j in range(self.input_depth):
#                 self.output[i] += correlate2d(self.input[j], self.kernels[i, j], mode=self.mode)[::self.stride, ::self.stride]
#         return self.output
#
#     def backward(self, output_gradient, learning_rate):
#         output_gradient = output_gradient.astype(np.float32)
#         # Initialize gradients
#         kernels_gradient = np.zeros_like(self.kernels, dtype=np.float32)
#         input_gradient = np.zeros_like(self.input, dtype=np.float32)
#
#         # Loop over each kernel (filter) and compute gradients
#         for i in range(self.depth):
#             for j in range(self.input_depth):
#                 for k in range(0, output_gradient.shape[1]):
#                     for l in range(0, output_gradient.shape[2]):
#                         current_position = (k * self.stride, l * self.stride)
#                         if current_position[0] + self.kernels_shape[2] <= self.input.shape[1] and current_position[1] + self.kernels_shape[3] <= self.input.shape[2]:
#                             input_slice = self.input[j, current_position[0]:current_position[0] + self.kernels_shape[2], current_position[1]:current_position[1] + self.kernels_shape[3]]
#                             kernels_gradient[i, j] += input_slice * output_gradient[i, k, l]
#
#                 for k in range(self.output_height):
#                     for l in range(self.output_width):
#                         current_position = (k * self.stride, l * self.stride)
#                         if current_position[0] + self.kernels_shape[2] <= self.input.shape[1] and current_position[1] + self.kernels_shape[3] <= self.input.shape[2]:
#                             input_gradient[j, current_position[0]:current_position[0] + self.kernels_shape[2], current_position[1]:current_position[1] + self.kernels_shape[3]] += self.kernels[i, j] * output_gradient[i, k, l]
#
#         # Check for NaN or Inf values before updating
#         if np.any(np.isnan(kernels_gradient)):
#             raise ValueError("NaN values encountered in kernels_gradient during backward pass")
#         if np.any(np.isnan(input_gradient)):
#             raise ValueError("NaN values encountered in input_gradient during backward pass")
#
#         # Update kernels and biases using the computed gradients
#         self.kernels -= learning_rate * kernels_gradient
#         self.biases -= learning_rate * output_gradient
#
#         return input_gradient


# class Convolutional(Layer):
#     def __init__(self, input_shape, kernel_shape, depth, stride=1, mode="valid"):
#         super().__init__()
#         input_depth, input_height, input_width = input_shape
#         kernel_height, kernel_width = kernel_shape
#         self.kernel_shape = kernel_shape   #  kernel shape and kernels shape are diff
#         self.depth = depth
#         self.input_shape = input_shape
#         self.input_depth = input_depth
#         self.stride = stride
#         self.mode = mode
#
#         if mode == "valid":
#             self.output_height = (input_height - kernel_height) // stride + 1
#             self.output_width = (input_width - kernel_width) // stride + 1
#         elif mode == "full":
#             self.output_height = (input_height + kernel_height - 1) // stride + 1
#             self.output_width = (input_width + kernel_width - 1) // stride + 1
#
#         self.output_shape = (depth, self.output_height, self.output_width)
#         self.kernels_shape = (depth, input_depth, kernel_height, kernel_width)
#         self.kernels = np.random.randn(*self.kernels_shape)
#         self.biases = np.random.randn(depth, self.output_height, self.output_width)
#
#     def forward(self, input):
#         # print("in convolution layer:", input.shape)
#         self.input = input
#         self.output = np.copy(self.biases)
#         for i in range(self.depth):
#             for j in range(self.input_depth):
#                 self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], self.mode)
#         print("nan in output conv: ", np.isnan(self.output).any())
#         return self.output
#
#
#     def backward(self, output_gradient, learning_rate):
#         # print("in backward convolution")
#         # print("output_gradient shape: ", output_gradient.shape)
#         # print("input shape: ", self.input.shape)
#         kernels_gradient = np.zeros(self.kernels_shape)
#         input_gradient = np.zeros(self.input_shape)
#
#         for i in range(self.depth):
#             for j in range(self.input_depth):
#                 kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], 'valid')
#                 input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], 'full')
#                 # dilated_gradient = dilate_matrix(output_gradient[i], self.stride - 1)
#                 # current_gradient = None
#                 #
#                 # # adding rows to balance output kernel gradient
#                 # if (self.input[j].shape[0] - output_gradient[i].shape[0]) / self.stride + 1 != self.kernel_shape[0]:
#                 #     current_gradient = add_zeros(dilated_gradient, self.stride - 1, 0)
#                 #
#                 # fully_dilated_gradient = None
#                 # if current_gradient is None:
#                 #     current_gradient = dilated_gradient
#                 # # adding cols to balance output kernel gradient
#                 # if (self.input[j].shape[1] - output_gradient[i].shape[1]) / self.stride + 1 != self.kernel_shape[1]:
#                 #     fully_dilated_gradient = add_zeros(current_gradient, self.stride - 1, 1)
#                 #
#                 # if fully_dilated_gradient is None:
#                 #     fully_dilated_gradient = current_gradient
#                 # # print("fully dilated gard shape: ", fully_dilated_gradient.shape)
#                 #
#                 # kernels_gradient[i, j] = correlate2d(self.input[j], fully_dilated_gradient, 1, "valid")
#                 #
#                 # padded_gradient_temp = add_zeros(fully_dilated_gradient, self.stride, 0)
#                 # padded_gradient = add_zeros(padded_gradient_temp, self.stride, 1)
#                 # # print("padded gard shape: ", padded_gradient.shape)
#                 # # print("convolve shape with stride: ", convolve2d(dilated_gradient, self.kernels[i, j], 1, "full").shape)
#                 # # print("convolve shape with kernel-1: ", convolve2d(padded_gradient, self.kernels[i, j], 1, "full").shape)
#                 #
#                 # input_gradient[j] += convolve2d(output_gradient[i], self.kernels[i, j], 1, "full")
#
#         self.kernels -= learning_rate * kernels_gradient
#         self.biases -= learning_rate * output_gradient
#         return input_gradient

class Convolutional(Layer):
    def __init__(self, input_shape, kernel_shape, depth, stride=1, mode="valid"):
        super().__init__()
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_shape[0] + 1, input_width - kernel_shape[1] + 1)
        self.kernels_shape = (depth, input_depth, kernel_shape[0], kernel_shape[1])
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        return self.output

    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient
