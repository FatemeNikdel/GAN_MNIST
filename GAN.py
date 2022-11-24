import numpy as np
import tensorflow as tf





#########   Variables    ########
initial_input = np.random.rand(100)
Batch_size = 32

#########  Real Data    ##########

def real_data():
    x,_, y,_ = tf.keras.datasets.mnist.load_data.mnist()

#########  Generator    ##########

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
                                        tf.keras.layers.Conv2DTranspose(1, (4,4), strides = (2,2), activation = "sigmoid", padding = "same"),
                                       ])

########## Descriminator ############

def descriminator():
    d_net = tf.keras.models.Sequential([
                                        tf.keras.layers.Conv2D(64, (3,3), strides = (2,2), padding = "same", input_shape = (28, 28, 1)),
                                        tf.keras.layers.LeakyReLU(alpha=0.2),
                                        tf.keras.layers.Dropout(0.4),
                                        tf.keras.layers.Conv2D(64, (3,3), strides=(2,2), padding = "same"),
                                        tf.keras.layers.LeakyReLU(alpha=0.2),
                                        tf.keras.layers.Dropout(0.4),
                                        tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dense(1, activation = "sigmoid")
                                       ])
    opt = tf.keras.optimizers.Adam(learning_rate = 0.002, beta_1 = 0.5)
    d_net.compile(optimizer = opt, loss="binary_crossentropy", metrics = ['accuracy'])
    d_net.summary()
    return d_net

descriminator()