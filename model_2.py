from typing import Callable

from loss_fns import CrossEntropyLoss
from loss_fns import MSE

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time

np.set_printoptions(linewidth=140)

LOSS_MAPPINGS = {'mse': MSE, 'binarycrossentropy': CrossEntropyLoss}


class Model:
    def __init__(self, loss: [str, Callable], layers = []):
        self.layers = layers if layers else []
        self.loss_fn = LOSS_MAPPINGS[loss.lower().replace(' ', '')]() if type(loss) == str else loss
            
    def add(self, layer: Callable):
        self.layers.append(layer)
    

    def _forward_prop(self, x):
        output = x
        
        for layer in self.layers:            
            output = layer.forward(output)

        return output
    
    
    def _backprop(self, loss_gradient):
        grad_of_layer_output = loss_gradient
        
        for layer in reversed(self.layers):
            grad_of_layer_output = layer.backward(grad_of_layer_output, self.lr)
            
    
    def get_accuracy(self, y_preds, y_true):

        # Count the number of correctly classified samples
        correct = np.sum(np.equal(np.argmax(y_true), np.argmax(y_preds)))
        print(np.argmax(y_true), np.argmax(y_preds))
        # Calculate accuracy
        accuracy = correct / y_true.shape[1]

        return accuracy    
    
    def train(self, train_x, train_y, epochs = 10, lr = 0.01, display_per_epochs = 1):
        self.lr = lr
        
        loss_over_epochs = []
        acc_over_epochs = []
        
        for epoch in range(epochs):
            loss = 0
            acc = 0
            
            timer_start = time()
            
            for x, y in zip(train_x, train_y):
                print(x.shape)
                # Reshaping X and Y
                y = y.reshape((y.shape[0], 1))
                
                if x.ndim == 2:
                    x = np.expand_dims(x, axis=0)
                print(x.shape)
                # Forward propogation
                y_preds = self._forward_prop(x)
                
                # Get loss and accuracy
                example_loss = self.loss_fn.forward(y_preds, y)
                example_acc = self.get_accuracy(y_preds, y)
                
                example_loss_grad = self.loss_fn.backward(y_preds, y)
                                
                # Backward propogation and gradients' updation
                self._backprop(example_loss_grad)
                                
                # Update the main loss and accuracy
                loss += example_loss
                acc += example_acc
            
            loss /= len(train_x)
            acc /= len(train_x)
            
            if epoch % display_per_epochs == 0:
                print(f"Epoch {epoch:3d}: Loss = {loss:4f}, Accuracy = {acc:4f}, Time = {(time()-timer_start):2f} sec per epoch")
            
            loss_over_epochs.append(loss)
            acc_over_epochs.append(acc)
        
    
    def predict(self, test_x):
        preds = self._forward_prop(test_x)
        
        return preds


# model = Model(loss='mse', layers=[])
