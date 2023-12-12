import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import objectives
from tensorflow.keras.datasets import mnist

# Load and preprocess MNIST data
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))

# Define VAE model
def build_vae(input_dim, intermediate_dim, latent_dim):
    x = Input(shape=(input_dim,))
    h = Dense(intermediate_dim, activation='relu')(x)
    
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

    def sampling(args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    decoder_h = Dense(intermediate_dim, activation='relu')
    decoder_mean = Dense(input_dim, activation='sigmoid')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)

    vae = Model(x, x_decoded_mean)

    xent_loss = input_dim * objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(xent_loss + kl_loss)

    vae.add_loss(vae_loss)
    vae.compile(optimizer='rmsprop')
    
    return vae

# Build VAE model
input_dim = x_train.shape[1]
vae = build_vae(input_dim, 256, 2)

# Train the VAE
vae.fit(x_train, shuffle=True, epochs=50, batch_size=128, validation_data=(x_test, None))

# Generate synthetic data using the trained VAE
n_samples = 10
noise = np.random.normal(0, 1, (n_samples, 2))
generated_data = vae.predict(noise)

# Visualize the generated data
plt.figure(figsize=(8, 4))
for i in range(n_samples):
    ax = plt.subplot(1, n_samples, i + 1)
    plt.imshow(generated_data[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
