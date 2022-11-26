import numpy as np
import tensorflow as tf

##########################################   Variables    ############################################################

Batch_size = 32



############################################  Generator    ###########################################################

def generator(data):
    n_node = 7 * 7 * 128      # The number of neurons of dense layer
    G_net = tf.keras.models.Sequential([
                                        tf.keras.layers.Dense(n_node, input_dim = data ),
                                        tf.keras.layers.LeakyReLU(alpha=0.2),
                                        tf.keras.layers.Reshape((7, 7, 128)),
                                        tf.keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding = "same"),
                                        tf.keras.layers.LeakyReLU(alpha=0.2),
                                        tf.keras.layers.Conv2D(64, (4,4), strides = (1,1), padding = "same"),
                                        tf.keras.layers.LeakyReLU(alpha=0.2),
                                        tf.keras.layers.Conv2DTranspose(1, (7,7), strides = (2,2), activation = "sigmoid", padding = "same"),
                                       ])
    return G_net

g_model = generator(100)

#######################################    Descriminator     ###########################################################

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

d_model = descriminator()
###########################################     GAN        ############################################################


def Gan( g_model , d_model):
    d_model.trainable = False                # in order not to train descriminator when generator is training 
    gan_net = tf.keras.models.Sequential([
        g_model,
        d_model
    ])

    opt = tf.keras.optimizers.Adam(leraning_rate = 0.0002, beta_1 = 0.5)
    gan_net.compile(loss = "binary_crossentropy", metrics = ['accuracy'])
    return gan_net

gan_model = Gan(g_model, d_model)

###########################################     Data      ############################################################

def Noise_data(noise_dim, Batch_size):
    x_input_Generator = np.random.randn(noise_dim * Batch_size)               
    x_input_Generator = x_input_Generator.reshape(Batch_size, noise_dim)
    return x_input_Generator

def fake_data(generator_model, noise_dim, Batch_size):
    x_input_Generator = Noise_data(noise_dim, Batch_size)
    X = generator_model.predict(x_input_Generator)
    y = np.zeros((noise_dim,1))
    return X, y

'''X , y = fake_data(g_model, 100, 32)'''

def Real_data():
    (x_train,_),(_,_) = tf.keras.datasets.mnist.load_data()
    x = np.reshape(x_train,(len(x_train), 28, 28, 1))
    x = x.astype("float32")
    x = x/255
    return x
real_data = Real_data()

def real_sample(real_data, batch_size):
    index = np.random.randint(0, len(real_data), batch_size)  # generate random index in size of batch data between [o , len real data]
    x = real_data(index)                                      # find the data of indexes
    y = np.ones((batch_size,1))                               # assign the label of real data which is 1

    return x , y

x_real , y_real = real_sample(real_data, Batch_size)
