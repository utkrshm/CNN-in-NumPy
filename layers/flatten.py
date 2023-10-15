import numpy as np
class Flatten:    
    def forward(self, layer_input):
        self.input_shape = layer_input.shape
        flattened_shape = int(np.prod(self.input_shape))
        return np.reshape(layer_input, (flattened_shape, 1))
    
    
    def backward(self, output_grad, lr):
        return np.reshape(output_grad, self.input_shape)