import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Reshape, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Generate random data
def generate_real_samples(n_samples):
    X = np.random.rand(n_samples) * 2 - 1  # Generate random values between -1 and 1
    y = np.ones((n_samples, 1))
    return X, y

# Generate noise as input for the generator
def generate_noise(n_samples, noise_dim):
    return np.random.randn(n_samples, noise_dim)

# Generator model
def build_generator(noise_dim):
    model = Sequential()
    model.add(Dense(10, input_dim=noise_dim, activation='relu'))
    model.add(Dense(1, activation='linear'))
    return model

# Discriminator model
def build_discriminator():
    model = Sequential()
    model.add(Dense(10, input_dim=1, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002), metrics=['accuracy'])
    return model

# Build and compile the GAN model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002))
    return model

# Train the GAN
def train_gan(generator, discriminator, gan, noise_dim, n_epochs=5000, n_batch=64):
    half_batch = n_batch // 2

    for epoch in range(n_epochs):
        # Train discriminator with real data
        real_data, real_labels = generate_real_samples(half_batch)
        d_loss_real = discriminator.train_on_batch(real_data, real_labels)

        # Train discriminator with generated data
        noise = generate_noise(half_batch, noise_dim)
        generated_data = generator.predict(noise)
        fake_labels = np.zeros((half_batch, 1))
        d_loss_fake = discriminator.train_on_batch(generated_data, fake_labels)

        # Train generator
        noise = generate_noise(n_batch, noise_dim)
        valid_labels = np.ones((n_batch, 1))
        g_loss = gan.train_on_batch(noise, valid_labels)

        # Print progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, D Loss Real: {d_loss_real[0]}, D Loss Fake: {d_loss_fake[0]}, G Loss: {g_loss}")

# Set random seed for reproducibility
np.random.seed(1000)

# Parameters
noise_dim = 5

# Build models
generator = build_generator(noise_dim)
discriminator = build_discriminator()
gan = build_gan(generator, discriminator)

# Train GAN
train_gan(generator, discriminator, gan, noise_dim)
