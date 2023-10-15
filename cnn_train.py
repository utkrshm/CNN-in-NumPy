""" 
INCOMPLETE!!

All that remains now, your old enemies.
    - Loss increasing, not decreasing. 
      The reason behind it, this time is that the position where you're defining your loss, is inaccurate.
      When it was defined outside the epochs, your loss increased for a while and then became stagnant.
      When it was being changed every epoch, your loss always remained stagnant. FIO why that is happening.
      Start with your loss function declarations in your Model class, and the Loss classes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras.datasets as datasets
from keras.utils import to_categorical

from layers.flatten import Flatten
from layers.dense import Dense
from layers.conv import Conv2D
from layers.max_pooling import MaxPooling

from loss_fns import *
from activations import *
from model import Model


ds = datasets.mnist
(x_train, y_train), (x_test, y_test) = ds.load_data()

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

(trnX, trnY), (valX, valY) = get_subsets(x_train, y_train, nums=[4, 9], n_samples=100)

layers = [
    Conv2D((28, 28), stride=2),
    Sigmoid(),
    MaxPooling(),
    Flatten(),
    Dense(98, 50),
    Sigmoid(),
    Dense(50, 10),
    Sigmoid(),
    Dense(10, 2),
    Softmax()
]

model = Model(loss=CrossEntropyLoss(), layers=layers)


trnX =  trnX / np.float64(255.0)

s_trnX = trnX[0]

# model.train(trnX, trnY, epochs=500, display_per_epochs=10, lr=0.1)
# print(trnY.sum(axis=1))

