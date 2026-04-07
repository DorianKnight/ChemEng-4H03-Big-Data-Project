# Author: Dorian Knight
# Created: April 2nd 2026
# Updated: April 5th 2026
# Description: Utilize Tensorflow machine learning model to classify bird calls

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Model input dimensions
discrete_frequencies = 128
frames_of_audio = 6000

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(discrete_frequencies, frames_of_audio)),
    keras.layers.Dense(128, activation=tf.nn.relu), # Hidden layer of 128 neurons using the rectified linear unit activation function
    keras.layers.Dense(128, activation=tf.nn.relu), # Hidden layer of 128 neurons using the rectified linear unit activation function
    keras.layers.Dense(2, activation=tf.nn.softmax) # Output layer picks the "highest probability outcome"
])

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])

def create_and_evaluate_model(req_epochs, train_data, train_label, test_data, test_label):
    model.fit(train_data, train_label, epochs=req_epochs)
    test_loss, test_accuracy = model.evaluate(test_data, test_label)
    print(f"Accuracy: {test_accuracy}")
    return test_accuracy

def main():
    required_epochs = 15

    # Load training data
    train_data = np.load("bird_call_train_data.npy")
    train_label = np.load("bird_call_train_label.npy")
    test_data = np.load("bird_call_test_data.npy")
    test_label = np.load("bird_call_test_label.npy")

    print(f"Training samples: {len(train_label)}")
    print(f"Testing samples: {len(test_label)}")

    model_accuracy = create_and_evaluate_model(required_epochs, train_data, train_label, test_data, test_label)
    print(f"Model accuracy is: {model_accuracy*100}")

main()