""" 
Note: 
    This layer does not compute the backpropagation separately.
    Backpropagation of a MaxPooling generally involves returning an image the shape of the original image,
    except that the backpropagated image has a 1 if the number represented the maximum value in that 
    particular pool, and a 0 otherwise.
    
    We compute this image (named self.switches) while in forward propagation because computation becomes 
    twice as expensive if forward and backward propagation are separately computed.
"""

import numpy as np

class MaxPooling:
    def __init__(self, pool_size: int = 2, stride: int = 2):
        self.pool_size = pool_size
        self.stride = stride
    
    
    def get_pools(self, image_channel):
        for row in range(0, image_channel.shape[0], self.stride):
            for column in range(0, image_channel.shape[1], self.stride):
                sub_window = image_channel[row:row + self.pool_size, column: column + self.pool_size]
                yield sub_window, (int(row/self.stride), int(column/self.stride)), (int(row/1), int(column/1))
    
    #                 (14, 14, 2)
    def forward(self, input_image):
        self.input_image_original_shape = input_image.shape
        if len(input_image.shape) == 2: 
            input_image = np.reshape(input_image, (*input_image.shape, 1))
        
        self.switches = np.zeros(input_image.shape)             # An array that duplicates the input image
        output_height = int((input_image.shape[0] - self.pool_size) / self.stride) + 1
        output_width = int((input_image.shape[1] - self.pool_size) / self.stride) + 1
        
        output = np.zeros((output_height, output_width, input_image.shape[2]))
        
        for channel_no in range(input_image.shape[2]):
            img_channel = input_image[:, :, channel_no]
            for pool, output_curr_loc, curr_input_loc in self.get_pools(img_channel):
                output_row_loc, output_col_loc = output_curr_loc
                input_row_loc, input_col_loc = curr_input_loc
                
                # Get the index at which the max value is stored, in 2D
                max_val_row_loc, max_val_col_loc = np.unravel_index(np.argmax(pool, axis=None), pool.shape)
                
                # Store the value 1 wherever the max value of the pool has come from
                self.switches[max_val_row_loc + input_row_loc, max_val_col_loc + input_col_loc, channel_no] = 1
                                
                # Store the max value
                output[output_row_loc, output_col_loc, channel_no] = np.amax(pool)
        
        return output
    
    def backward(self, output_grad, lr):
        return self.switches.reshape(self.input_image_original_shape)
            #    Shape -      (14, 14, 2)