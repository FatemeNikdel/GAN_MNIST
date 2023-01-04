import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


############################################  Variable     ###########################################################
Batch_size = 256
noise_dim = 100
epochs = 5
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
    #d_net.summary()
    return d_net

d_model = descriminator()
###########################################     GAN        ############################################################


def Gan( g_model , d_model):
    d_model.trainable = False                # in order not to train descriminator when generator is training 
    gan_net = tf.keras.models.Sequential([
        g_model,
        d_model
    ])

    opt = tf.keras.optimizers.Adam(learning_rate = 0.0002, beta_1 = 0.5)
    gan_net.compile(loss = "binary_crossentropy", metrics = ['accuracy'])
    return gan_net

gan_model = Gan(g_model, d_model)

###########################################     Data      ############################################################

def Noise_data(noise_dim, batch_size):
    x_input_Generator = np.random.randn(noise_dim * batch_size)               
    x_input_Generator = x_input_Generator.reshape(batch_size, noise_dim)
    return x_input_Generator

def fake_data(g_model, noise_dim, batch_size):
    x_input_Generator = Noise_data(noise_dim, batch_size)
    X = g_model.predict(x_input_Generator)
    y = np.zeros((batch_size,1))
    return X, y

x_fake , y_fake = fake_data(g_model, 100, 32)

def Real_data():
    (x_train,_),(_,_) = tf.keras.datasets.mnist.load_data()
    x = np.reshape(x_train,(len(x_train), 28, 28, 1))
    x = x.astype("float32")
    x = x/255
    return x
real_data = Real_data()

def real_sample(real_data, batch_size):
    index = np.random.randint(0, real_data.shape[0], batch_size)  # generate random index in size of batch data between [o , len real data]
    x = real_data[index]                                          # find the data of indexes
    y = np.ones((batch_size,1))                                   # assign the label of real data which is 1

    return x , y

x_real , y_real = real_sample(real_data, Batch_size)

##############################################    show result     ####################################################

def save_plot(examples, epoch, n=10):

	for i in range(n * n):
		plt.subplot(n, n, 1 + i)
		plt.axis('off')
		plt.imshow(examples[i, :, :, 0], cmap='gray_r')
	filename = 'generated_plot_e%03d.png' % (epoch+1)
	plt.savefig(filename)
	plt.close()

def summarize_performance(epoch, g_model, d_model, dataset, noise_dim, batch_size=100):
    X_real, y_real = real_sample(dataset, batch_size)
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    x_fake, y_fake = fake_data(g_model, noise_dim, batch_size)
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    print(f'>Accuracy real: {acc_real*100}, fake: {acc_fake*100}')
    save_plot(x_fake, epoch)
    filename = f'generator_model_{epoch + 1}.h5'
    g_model.save(filename)

##############################################       Train         #####################################################

def train(d_model, g_model, gan_model, real_data, noise_dim, epochs, batch_size):
    batch_per_epoch = int(real_data.shape[0] / batch_size)
    half_batch = int(batch_size/2)
    for i in range(epochs):
        for j in range(batch_per_epoch):
            # train descriminator
            x_real, y_real = real_sample(real_data, half_batch)
            x_fake , y_fake = fake_data(g_model, noise_dim, half_batch)
            X , y = np.vstack((x_real, x_fake)), np.vstack((y_real, y_fake))
            d_loss , _ = d_model.train_on_batch(X, y)
            # train gan
            x_gan = Noise_data(noise_dim, batch_size)
            y_gan = np.ones((batch_size , 1))
            g_loss, _ = gan_model.train_on_batch(x_gan, y_gan)
            # show results
            print( f"===>>>>{i+1},{j+1}/{batch_per_epoch}, d = {d_loss:.3f}, g = {g_loss:.3f}")
            if (i+1) % 10 == 0:
                summarize_performance(i, g_model, d_model, real_data, noise_dim)
train(d_model, g_model, gan_model, real_data, noise_dim, epochs, Batch_size)
