import os
import h5py
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, Embedding, Concatenate
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

# Carregar o conjunto de dados MNIST
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 127.5 - 1.0  # Normalização para o intervalo [-1, 1]

# Tamanho do vocabulário (números de 0 a 9)
vocab_size = 10

# Dimensão do vetor de ruído
noise_dim = 100

# Modelo GAN condicionado por texto
def build_generator():
    model = Sequential()
    model.add(Dense(256, input_dim=noise_dim))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(28 * 28, activation='tanh'))  # Saída com 28x28=784 unidades
    model.add(Reshape((28, 28, 1)))  # Redimensionar para o tamanho da imagem MNIST
    return model

def build_discriminator():
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28, 1)))  # Flatten para imagem MNIST
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

def build_gan(generator, discriminator):
    noise = tf.keras.layers.Input(shape=(noise_dim,))
    label = tf.keras.layers.Input(shape=(1,), dtype='int32')
    label_embedding = Embedding(vocab_size, noise_dim)(label)
    flat_embedding = Flatten()(label_embedding)
    model_input = Concatenate()([noise, flat_embedding])
    generated_image = generator(model_input)
    discriminator_output = discriminator(generated_image)
    gan = tf.keras.models.Model([noise, label], discriminator_output)
    return gan

generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
discriminator.trainable = False
gan = build_gan(generator, discriminator)

# Treinamento da GAN
def train_gan(epochs, batch_size, save_interval):
    for epoch in range(epochs):
        for _ in range(len(x_train) // batch_size):
            batch_images = x_train[np.random.randint(0, x_train.shape[0], batch_size)]
            noise = np.random.normal(0, 1, (batch_size, noise_dim))
            labels = np.random.randint(0, vocab_size, batch_size).reshape(-1, 1)
            generated_images = generator.predict([noise, labels])
            discriminator_loss_real = discriminator.train_on_batch(batch_images, np.ones((batch_size, 1)))
            discriminator_loss_generated = discriminator.train_on_batch(generated_images, np.zeros((batch_size, 1)))
            discriminator_loss = 0.5 * (discriminator_loss_real + discriminator_loss_generated)
            noise = np.random.normal(0, 1, (batch_size, noise_dim))
            labels = np.random.randint(0, vocab_size, batch_size).reshape(-1, 1)
            gan_loss = gan.train_on_batch([noise, labels], np.ones((batch_size, 1)))

        if (epoch + 1) % save_interval == 0:
            save_generated_image(epoch + 1, generator)
            
            # Salvar o modelo GAN e os pesos do gerador e discriminador
            generator.save('generator_model.h5')
            discriminator.save('discriminator_model.h5')
            gan.save('gan_model.h5')

def save_generated_image(epoch, generator, examples=1, filename="gan_generated_image.png"):
    noise = np.random.normal(0, 1, (examples, noise_dim))
    labels = np.array(range(vocab_size))
    generated_images = generator.predict([noise, labels])
    generated_images = (generated_images + 1) / 2.0  # Ajuste para o intervalo [0, 1]

    plt.figure(figsize=(6, 6))
    for i in range(examples):
        plt.subplot(examples, 1, i+1)
        plt.imshow(generated_images[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Treinar a GAN
train_gan(epochs=10000, batch_size=64, save_interval=100)
