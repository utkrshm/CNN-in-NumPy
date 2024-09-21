import numpy as np
from scipy.signal import correlate2d, convolve2d

""" 
Input dimensions for this function: (n_C, L, W) where 
                                                            n_C = number of image channels,
                                                              L = height of each image, and
                                                              W = width of each image.
                                                              
The function takes 4 other arguments, of number of filters in the layer, the size of each kernel, and the stride of the kernel and the padding of the image

The output of the forward propagation layer, has the shape: (n_K, out_L, out_W) where
                                                            n_K = number of filters in each layer,
                                                            out_L = ⌊ (L - f + 2p) / s + 1 ⌋ 
                                                            out_W = ⌊ (W - f + 2p) / s + 1 ⌋     
                                                            and f is the size of the kernels, s is the stride and p in the padding on each side

The shape of the filters in the layer will be: (n_K, n_C, K, K) where K is the size of each kernel.

The backward propagation function returns the input gradient with the shape: (K, out_L, out_W)                                               
"""

class Conv2D:
    def __init__(self, input_shape: tuple, num_kernels: int = 2, kernel_size: int = 2, stride: int = 1, pad: int = 0):
        
        n_channels, in_length, in_width = input_shape
        
        assert in_length == in_width, "Image input dimensions not equal."
        
        self.num_kernels = num_kernels
        self.n_channels = n_channels
        self.input_shape = input_shape
        
        self.stride = stride
        self.pad = pad
        self.kernel_size = kernel_size
        
        output_length = int((in_length - kernel_size + 2 * pad) / 2 + 1)
        output_width = int((in_width - kernel_size + 2 * pad) / 2 + 1)
        
        self.output_shape = (num_kernels, output_length, output_width)
        self.kernel_shape = (num_kernels, n_channels, kernel_size, kernel_size)
        
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

        for kernel_num in range(self.num_kernels):
            for channel in range(self.n_channels):
                for sub_window, output_row_loc, output_col_loc in self.get_sub_windows(self.input_image[channel]):

                    output_img[kernel_num, output_row_loc, output_col_loc] = correlate2d(sub_window, self.kernel_weights[kernel_num, channel], 'valid')                    
                    output_img[kernel_num] += self.kernel_biases[kernel_num]

        return output_img
        

    def backward(self, output_grad, lr):
        dkW = np.zeros(self.kernel_shape)
        input_grads = np.zeros(self.input_shape)

        for kernel_num in range(self.num_kernels):
            for channel in range(self.n_channels):
                print("Rotated shape:", np.rot90(self.kernel_weights[kernel_num, channel], 2).shape)

                for sub_window, output_row_loc, output_col_loc in self.get_sub_windows(self.input_image[channel]):

                    dkW[kernel_num, channel] = correlate2d(sub_window, self.kernel_weights[kernel_num, channel], 'valid')
                    # input_grads[channel, output_row_loc, output_col_loc] = convolve2d(output_grad[kernel_num], np.rot90(self.kernel_weights[kernel_num, channel], 2), 'full')
                    
                    print(convolve2d(output_grad[kernel_num], np.rot90(self.kernel_weights[kernel_num, channel], 2), 'full').shape)
                    raise Exception

        self.kernel_weights -= lr * dkW
        self.kernel_biases -= lr * output_grad

        return input_grads


import torch
torch.nn.Conv2d