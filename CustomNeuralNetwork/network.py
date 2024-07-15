import warnings

import numpy as np


def clip_gradients(gradients, threshold):
    norm = np.linalg.norm(gradients)
    if norm > threshold:
        gradients = gradients * (threshold / norm)
    return gradients


def predict(network, input):
    output = input
    for layer in network:
        temp = output
        output = layer.forward(output)
        if np.isnan(output).any() or np.isinf(output).any() or np.isneginf(output).any():
            print(f"forward::: inf/nan occured in {layer}, output shape: {output.shape}")
            print("input ", temp)
            print("output ", output)
            # print("weight ", layer.weights.shape, layer.weights)
            # print("bias ", layer.bias)
            # print("intermediate:", np.dot(layer.weights, temp))
            exit()
    return output


def train(network, loss, loss_prime, x_train, y_train, epochs=1000, learning_rate=0.01, verbose=True):
    for e in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            # forward
            output = predict(network, x)  #gives final prediction at the last layer
            # error
            error += loss(y, output)
            # backward
            grad = loss_prime(y, output)
            if np.isnan(grad).any() or np.isinf(grad).any() or np.isneginf(grad).any():
                print(f"backward::: inf/nan occured in grad of output, grad shape: {grad.shape}")
                exit()
            for layer in reversed(network):
                prev = grad
                grad = layer.backward(grad, learning_rate)
                # print("norm of current grad: ", np.linalg.norm(grad))
                grad = np.clip(grad, -500, 500)
                if np.isnan(grad).any() or np.isinf(grad).any() or np.isneginf(grad).any():
                    print(np.isnan(grad).any(), np.isinf(grad).any(), np.isneginf(grad).any())
                    print(f"backward::: inf/nan occured in grad of {layer}, grad shape: {grad.shape}")
                    print("prev grad: ", prev)
                    print("grad: ", grad)
                    exit()

        error /= len(x_train)
        if verbose:
            print(f"{e + 1}/{epochs}, error={error}")
            # print("\n \n")
