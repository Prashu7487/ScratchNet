import time
import pandas as pd
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from CustomNeuralNetwork.dense import Dense
from CustomNeuralNetwork.convolutional import Convolutional
from CustomNeuralNetwork.reshape import Reshape
from CustomNeuralNetwork.activations import ReLU, Tanh, Sigmoid, Softmax
from CustomNeuralNetwork.losses import mse, mse_prime
from CustomNeuralNetwork.network import train, predict
from CustomNeuralNetwork.poolfilters import MaxPooling2D
from CustomNeuralNetwork.normalize import Normalize


# Data loading and preprocessing

def load_images_from_directory(img_path, data):
    images = []
    ages = []
    genders = []

    # Randomly sample 1500 records from the data
    data = data.sample(50, random_state=42)

    for index, row in data.iterrows():
        ID, age, gender = row['id'], row['boneage'], row['male']

        # Corrected the image path construction
        img_file_path = os.path.join(img_path, f'{ID}.png')

        if os.path.exists(img_file_path):
            img = Image.open(img_file_path)
            # print(img.size)
            img = img.resize((256, 256))
            img_array = np.array(img)
            img_array = img_array.reshape((1, 256, 256))  #depth of input 1
            # print("img shape:", img_array.shape)
            images.append(img_array)
            ages.append(age)
            genders.append(1 if gender == 'TRUE' else 0)

    return np.array(images), np.array(ages), np.array(genders)


img_path = 'Datasets/bone-age/boneage-training-dataset/boneage-training-dataset'

data = pd.read_csv(r'Datasets/bone-age/boneage-training-dataset.csv')

images, ages, genders = load_images_from_directory(img_path, data)


# Split the data into training and testing sets
train_images, test_images, train_ages, test_ages = train_test_split(images, ages, test_size=0.2, random_state=42)
# print(train_images.shape)
train_images = train_images/255.0   # normalize images in the array

# random outlier img to test the difference in output
img = Image.open(r"C:\Users\pm748\OneDrive\Pictures\Screenshots\Screenshot 2024-05-23 231148.png")
img = img.resize((256, 256))
img_array = np.array(img)
img_array = img_array[:,:,2]
img_array = img_array.reshape((1, 256, 256))
test_images = list(test_images)
test_images.append(img_array)
test_images = np.array(test_images)

test_ages = list(test_ages)
test_ages.append(23)
test_ages = np.array(test_ages)


# yet to correct the model, confirm about Normalization, batch size, optimizers
model = [
    Convolutional((1, 256, 256), (10, 10), 1, stride=1, mode='valid'),  # Output: (1, 247, 247)
    MaxPooling2D((5, 5), stride=1),  # Output: (1, 243, 243)
    Normalize(),
    Convolutional((1, 243, 243), (10, 10), 1, stride=1, mode='valid'),  # Output: (1, 234, 234)
    MaxPooling2D((5, 5), stride=1),  # Output: (1, 230, 230)
    Normalize(),
    Convolutional((1, 230, 230), (10, 10), 1, stride=1, mode='valid'),  # Output: (1, 221, 221)
    MaxPooling2D((5, 5), stride=1),  # Output: (1, 217, 217)
    Normalize(),
    Reshape((1, 217, 217), (217*217, 1)),
    Dense(217*217, 1000),  # Input size: 217*217, Output size: 1000
    Normalize(),
    # ReLU(),  # Activation function
    Dense(1000, 100),  # Input size: 1000, Output size: 100
    Normalize(),
    ReLU(),  # Activation function
    Dense(100, 1),  # Input size: 100, Output size: 1
    # ReLU()
]


start_time = time.time()
print("Training starts...")
train(model, mse, mse_prime, train_images, train_ages, epochs=100, learning_rate=0.01)
end_training_time = time.time()

print("Testing starts...")

y_pred = []
y_true = []
for x, y in zip(test_images, test_ages):
    output = predict(model, x)
    print(f"pred: {output}, true: {y}")
    y_pred.append(output)
    y_true.append(y)

mse = mse(np.array(y_true), np.array(y_pred))
print(f"MSE: {mse}")

end_testing_time = time.time()
print("Total training time: ", end_training_time-start_time)
print("total testing time: ", end_testing_time-end_training_time)

