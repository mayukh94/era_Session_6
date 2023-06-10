
<!-- markdownlint-disable MD030 -->

# ERA Session 6 Assignment - Mayukh

<a href="https://theschoolof.ai/#programs">Learn about the program here</a>

## 📝 Overview

### Neural Network for MNIST dataset

Developed a neural network utilizing convolutions, max pooling, batch normalization, dropout and activation functions in order to recognize hand written digits from the MNIST dataset.

The neural network achieves 99.45% accuracy within 20 epochs and utilizes 19,668 parameters to do so.

The crux of this application is contained within the Model class, the rest of the application deals with the nitty gritty of setting up data loaders, train and test methods, optimizers etc.

### Model 

The model in question is made up of several layers, they are as follows:

## Layer 1

The first layer (layer1) is defined as a sequential composition of several operations

- nn.Conv2d(1, 8, 3, padding=1) creates a 2-dimensional convolutional layer with 1 input channel, 8 output channels, and a kernel size of 3x3. Padding of 1 is applied to ensure the spatial dimensions of the input and output tensors are the same.
- nn.ReLU() applies the rectified linear unit activation function element-wise to the output of the previous convolutional layer.
- nn.BatchNorm2d(8) applies batch normalization to the previous layer's output along the channel dimension (2nd dimension).
- nn.Conv2d(8, 16, 3, padding=1) creates another convolutional layer with 8 input channels, 16 output channels, and a kernel size of 3x3.
- nn.ReLU() applies the rectified linear unit activation function to the output of the previous convolutional layer.
- nn.BatchNorm2d(16) applies batch normalization.
- nn.MaxPool2d(2, 2) performs 2x2 max pooling with a stride of 2, reducing the spatial dimensions of the input tensor by half.
- nn.Conv2d(16, 8, 1) creates a 1x1 convolutional layer with 16 input channels and 8 output channels.
- nn.Dropout(0.05) applies dropout regularization by randomly setting 5% of the elements of the previous layer's output to zero during training.

## Layer 2

The second layer (layer2) is defined in a similar manner to layer1:

- nn.Conv2d(8, 16, 3) creates a convolutional layer with 8 input channels, 16 output channels, and a kernel size of 3x3.
- nn.ReLU() applies the rectified linear unit activation function.
- nn.BatchNorm2d(16) applies batch normalization.
- nn.Conv2d(16, 32, 3) creates a convolutional layer with 16 input channels, 32 output channels, and a kernel size of 3x3.
- nn.ReLU() applies the rectified linear unit activation function.
- nn.BatchNorm2d(32) applies batch normalization.
- nn.Conv2d(32, 32, 3) creates a convolutional layer with 32 input channels, 32 output channels, and a kernel size of 3x3.
- nn.ReLU() applies the rectified linear unit activation function.
- nn.BatchNorm2d(32) applies batch normalization.
- nn.MaxPool2d(2, 2) performs 2x2 max pooling with a stride of 2.
- nn.Dropout(0.05) applies dropout regularization.

## Convolutional Layer 3

The third convolutional layer (conv3) consists of the following operations:

- nn.Conv2d(32, 10, 3) creates a convolutional layer with 32 input channels, 10 output channels, and a kernel size of 3x3.
- nn.ReLU() applies the rectified linear unit activation function.
- nn.BatchNorm2d(10) applies batch normalization.
- nn.MaxPool2d(2, 2) performs 2x2 max pooling with a stride of 2.
- nn.Dropout(0.05) applies dropout regularization.

## Convolutional Layer 4

The fourth convolutional layer (conv4) consists of a single operation:

- nn.Conv2d(10, 10, kernel_size=1, stride=1, padding=0) creates a 1x1 convolutional layer with 10 input channels and 10 output channels. No padding is applied.

## Forward Pass

The forward method defines the forward pass computation of the model. Given an input tensor x, it applies the defined layers in a sequential manner:

- x = self.layer1(x) applies layer1 to the input tensor, producing an intermediate output.
- x = self.layer2(x) applies layer2 to the intermediate output, producing another intermediate output.
- x = self.conv3(x) applies conv3 to the intermediate output, producing yet another intermediate output.
- x = self.conv4(x) applies conv4 to the previous intermediate output, producing the final convolutional output.
- x = x.view(x.size(0), -1) reshapes the output tensor to have a shape of (batch_size, -1), effectively flattening the tensor while preserving the batch dimension.
- x = F.log_softmax(x, dim=1) applies the logarithm of the softmax function along the second dimension of the tensor, which represents the class probabilities.
- The resulting tensor x is returned as the output of the forward pass.

## 🙌 Contributing

No Contributions are allowed unfortunately :(
