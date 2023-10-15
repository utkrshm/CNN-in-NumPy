import numpy as np
from scipy.signal import correlate2d, convolve2d

""" 
Note: This is a 2D Convolution layer, only accepts images with one colour channel. 
      This works for our use case, as we are testing this with MNIST, which only has the grayscale channel.
      This layer would also not work, if this is the second Convolution layer used in our model.
"""

class Conv2D:
    def __init__(self, input_shape: tuple, num_kernels: int = 2, kernel_size: int = 2, stride: int = 1, pad: int = 0):
        assert len(input_shape) == 2, "Image must have exactly two dimensions!"
        
        in_length, in_width = input_shape
        
        self.num_kernels = num_kernels
        self.input_shape = input_shape
        
        self.stride = stride
        self.pad = pad
        self.kernel_size = kernel_size
        
        output_dim = int((in_length - kernel_size + 2 * pad) / 2) + 1
        
        self.output_shape = (output_dim, output_dim, num_kernels)
        self.kernel_shape = (kernel_size, kernel_size, num_kernels)
        
        self.kernel_weights = np.random.randn(*self.kernel_shape) * 0.01
        self.kernel_biases = np.random.randn(*self.output_shape) * 0.01
        
    
    def get_sub_windows(self, image_channel):
        for row in range(0, self.output_shape[0], self.stride):
            for column in range(0, self.output_shape[1], self.stride):
                sub_window = image_channel[row:row + self.kernel_size, column: column + self.kernel_size]
                yield sub_window, int(row / self.stride), int(column / self.stride)

    
    def forward(self, input_image):
        output_img = np.zeros(self.output_shape)
        
        if self.pad > 0:
            input_image = np.pad(input_image, ((self.pad, self.pad), (self.pad, self.pad)), 'constant')

        self.input_image = input_image
        
        for sub_window, output_row_loc, output_col_loc in self.get_sub_windows(self.input_image):
            output_img[output_row_loc, output_col_loc] = np.sum(sub_window * self.kernel_weights, axis=(0, 1))
            output_img += self.kernel_biases
        return output_img
        

    def backward(self, output_grad, lr):
        dkW = np.zeros(self.kernel_shape)
        
        for sub_window, output_row_loc, output_col_loc in self.get_sub_windows(self.input_image):
            for n in range(self.num_kernels):
                dkW[:, :, n] += output_grad[output_row_loc, output_col_loc, n] * sub_window

        self.kernel_weights -= lr * dkW
        self.kernel_biases -= lr * output_grad
        
        # We don't need to return anything here, since this will be the last layer of our neural network.
        # The input gradients of this layer are useless, as the input of this layer is our MNIST images.
        return None