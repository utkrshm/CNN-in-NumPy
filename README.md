This is the code I wrote for a Convolutional Neural Network from scratch, purely in NumPy and using only arrays. The only exceptions are when I used Keras to load the MNIST dataset and to turn my output, into a categorical numpy array. Other than this, the math and everything else was self-coded, with inspiration for some functions, taken from various GitHub repos.

This model is built like how Tensorflow/Keras's Sequential models work, taking in a list of layer classes and computing the forward propagation and backpropgation algorithms within the layers, and storing the parameters within the layers themselves.

----

## Architecture of the CNN made

Training data: (160, 28, 28)

Per Image: (28, 28)

Per image network architecture:
- Convolution - (28, 28) to (14, 14, 2)
  - Filter size: (2x2)
  - Stride: 2
  - No of kernels = no of digits to be recognized (in this case, 2)
- Activation - Sigmoid
- MaxPooling - (14, 14, 2) to (7, 7, 2)
  - Stride: 1
  - Filter size: (2x2)
  - Pad: None
- Reshape - (7, 7, 2) to (98, 1) (flatten operation)
- Dense - 98 to 35
- Activation - Softmax
- Dense - 35 to 2
- Activation - Softmax
- Loss - Cross Entropy
