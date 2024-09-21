import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical

from layers.flatten import Flatten
from layers.dense import Dense
from layers.conv_2 import Conv2D
from layers.max_pooling import MaxPooling
from loss_fns import *
from activations import *
from model_2 import Model

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Final and V3 of the get_subsets function - This one returns the training and validation sets' inputs and outputs
def get_subsets(X, Y, nums: list, n_samples: int, val_pct = 0.2):
    assert val_pct < 0.4, "Validation Percentage must not be greater than 0.4!"
    
    trn_indices = np.array([], dtype=int)
    val_indices = np.array([], dtype=int)
    for num in nums:
        # Get all indices containing the images of that particular number
        indices = np.where(Y == num)[0]
        
        # Randomly shuffling the indices
        np.random.shuffle(indices)
        
        # Getting the first N images from the indices
        indices = indices[:n_samples]
        
        # Get the number of training images for these indices
        num_images = len(indices)
        num_train_images = int(num_images * (1 - val_pct))
        
        # Add the training and validation images to their respective lists. If X is the number of training images,
        # then 0-X are the training images and X-(-1) are the validation images.
        trn_indices = np.concatenate((trn_indices, indices[:num_train_images]))
        val_indices = np.concatenate((val_indices, indices[num_train_images:]))
        
    # Shuffling the two arrays properly to make sure that the images mix properly
    np.random.shuffle(trn_indices)
    np.random.shuffle(val_indices)
    
    # Changing the outputs to be categorical, and making sure that we only carry the relevant numbers out of the function.
    Y = to_categorical(Y)
        
    Y = Y[:, nums]
    
    # Get and return the validation sets, and training sets
    #               Training set                      Validation set
    return (X[trn_indices], Y[trn_indices]), (X[val_indices], Y[val_indices])


# HYPERPARAMETERS
CLASSES = [2, 3, 4, 5, 8, 9]
N_SAMPLES = 100
lr = 5


# # Actual training
(trnX, trnY), (valX, valY) = get_subsets(x_train, y_train, nums=CLASSES, n_samples=N_SAMPLES)

layers = [
    Conv2D((1, 28, 28), stride=2),
    Sigmoid(),
    MaxPooling(),
    Flatten(),
    Dense(98, 50),
    Sigmoid(),
    Dense(50, 10),
    Sigmoid(),
    Dense(10, len(CLASSES)),
    Softmax()
]

model = Model(loss=CrossEntropyLoss(), layers=layers)


trnX =  trnX / np.float64(255.0)

s_trnX = trnX[0]

model.train(trnX, trnY, epochs=50, display_per_epochs=1, lr=lr)
