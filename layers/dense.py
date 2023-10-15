import numpy as np

class Dense:
    def __init__(self, n_input, n_output):
        self.input_n = n_input
        self.output_n = n_output
        
        self.weights = np.random.randn(self.output_n, self.input_n) * 0.01
        self.biases = np.random.randn(self.output_n, 1) * 0.01
        
        self.num_params = self.weights.size + self.biases.size
    
    
    def forward(self, layer_input):
        self.input = layer_input
        return (self.weights @ self.input) + self.biases
    
    
    def backward(self, output_grad, lr):
        # Calculating the gradients
        dW = output_grad @ self.input.T
        db = np.sum(output_grad, axis=1, keepdims=True)
        
        input_grads = self.weights.T @ output_grad
        
        # Updating the parameters
        self.weights -= lr * dW
        self.biases -= lr * db
        
        return input_grads
        