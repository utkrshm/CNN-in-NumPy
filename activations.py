import numpy as np

class Sigmoid:
    def _sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
    
    
    def forward(self, layer_input):
        self.input = layer_input
        return self._sigmoid(self.input)
    
    
    def backward(self, output_grad, lr):
        dZ = self._sigmoid(self.input) * (1 - self._sigmoid(self.input))
        return output_grad * dZ
    

class ReLU:
    def _relu(self, Z):
        return np.maximum(Z, 0)
    
    
    def forward(self, layer_input):
        self.input = layer_input
        
        return self._relu(self.input)

    
    def backward(self, output_grad, lr):
        dZ = (self.input > 0).astype(int)
        return output_grad * dZ
    

class Softmax:
    def _softmax(self, Z):
        return np.exp(Z) / np.sum(np.exp(Z))
    
    
    def forward(self, layer_input):
        self.input = layer_input
        return self._softmax(layer_input)
    
    
    def backward(self, output_grad, lr):
        n = self.input.size
        activated_input = self._softmax(self.input)
                
        return np.dot(
            ( activated_input * ( np.identity(n) - activated_input.T ) ),
            output_grad
        )
