
# GAN for generating fake MNIST digits

This code generates fake MNIST digits using a GAN (Generative Adversarial Network) model implemented in TensorFlow. The generator model is trained to generate images that are similar to the MNIST dataset, and the discriminator model is trained to distinguish between the real and fake images. The two models are then combined to create a GAN model that can generate new, fake images.

## Requirements
This code requires the following packages to be installed:

- numpy
- tensorflow
- matplotlib

## Usage
1. Download the code and save it to a file (e.g. gan_mnist.py).
2. Run the code using python gan_mnist.py.

## Output
The code outputs the following:

- The generator model is saved to a file named generator_model_[epoch].h5.
- A plot of generated images is saved to a file named generated_plot_e[epoch].png.

## Parameters
The following parameters can be adjusted:

- Batch_size: The batch size for training the models.
- noise_dim: The dimension of the noise vector used as input to the generator.
- epochs: The number of epochs to train the models for.

