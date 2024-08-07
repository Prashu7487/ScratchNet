import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from ScratchNet.dense import Dense
from ScratchNet.activations import Sigmoid
from ScratchNet.losses import binary_cross_entropy, binary_cross_entropy_prime
from ScratchNet.network import train, predict

# Load Iris dataset
iris = load_iris()
X = iris.data[:, :2]  # Use only sepal length and sepal width
y = (iris.target == 0).astype(int).reshape(-1, 1)  # Binary classification: Is the flower Iris Setosa?

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reshape data for the network
X = np.expand_dims(X, axis=2)
Y = np.expand_dims(y, axis=2)

# Define the network architecture
network = [
    Dense(2, 4),
    Sigmoid(),
    Dense(4, 1),
    Sigmoid()
]

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train the network
train(network, binary_cross_entropy, binary_cross_entropy_prime, X_train, Y_train, epochs=1000, learning_rate=0.01)  #try to 10000 epochs

# Predict and evaluate
correct_predictions = 0
total_predictions = 0

for x, y in zip(X_test, Y_test):
    output = predict(network, x)
    pred = 1 if output > 0.5 else 0
    true_value = int(y.item())  # Ensure y is converted to a scalar
    if pred == true_value:
        correct_predictions += 1
    total_predictions += 1
    print('pred:', pred, '\ttrue:', true_value)

# Calculate and print accuracy
accuracy = correct_predictions / total_predictions
print(f'Accuracy: {accuracy:.2f}')
