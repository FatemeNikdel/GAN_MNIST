import numpy as np
import tensorflow as tf



#Generator Structure  (output = (28*28*1))
# Inpit = noise 100D - (7*7*256) - (14*14*128) - 


initial_input = np.random.rand(100)

def generator():
    n_node = 7 * 7 * 128      # The number of neurons of dense layer
    G_net = tf.keras.models.Sequential([
                                        tf.keras.layers.Dense(n_node, input_dim = initial_input ),
                                        tf.keras.layers.LeakyReLU(alpha=0.2),
                                        tf.keras.layers.Reshape((7, 7, 128)),
                                        tf.keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding = "same"),
                                        tf.keras.layers.LeakyReLU(alpha=0.2),
                                        tf.keras.layers.Conv2D(64, (4,4), strides = (1,1), padding = "same"),
                                        tf.keras.layers.LeakyReLU(alpha=0.2),
                                        tf.keras.layers.Conv2DTranspose(1, (4,4), strides=(2,2), activation = "sigmoid", padding = "same"),
                                       ])

def real_data():
    x,_, y,_ = tf.keras.datasets.mnist.load_data()