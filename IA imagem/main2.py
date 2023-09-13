import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Flatten, LeakyReLU, Conv2D, UpSampling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import os

# Gerar um conjunto de dados fictício para teste
def generate_fake_data(num_samples, image_size):
    return np.random.rand(num_samples, *image_size) * 2 - 1  # Imagens aleatórias no intervalo [-1, 1]

# Tamanho das imagens fictícias
image_size = (128, 128, 3)  # Altere para (128, 128, 1) se estiverem em escala de cinza

# Número de amostras fictícias para teste
num_samples = 1000

# Conjunto de dados fictício para teste
x_train = generate_fake_data(num_samples, image_size)

# Diretório para salvar modelos e imagens geradas
os.makedirs("gan_models", exist_ok=True)
os.makedirs("gan_generated_images", exist_ok=True)

# Definir o gerador
def build_generator():
    model = Sequential()
    model.add(Dense(256, input_dim=100))
    model.add(LeakyReLU(0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(0.2))
    model.add(Dense(1024))
    model.add(LeakyReLU(0.2))
    model.add(Dense(np.prod(image_size), activation='tanh'))
    model.add(Reshape(image_size))
    return model

# Definir o discriminador
def build_discriminator():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=image_size))
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

# Criar modelos do gerador e do discriminador
generator = build_generator()
discriminator = build_discriminator()

# Compilar o discriminador
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

# Construir a GAN
discriminator.trainable = False
gan_input = tf.keras.layers.Input(shape=(100,))
x = generator(gan_input)
gan_output = discriminator(x)
gan = tf.keras.models.Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# Função para treinar a GAN
def train_gan(epochs, batch_size):
    for epoch in range(epochs):
        for _ in range(x_train.shape[0] // batch_size):
            noise = tf.random.normal((batch_size, 100))
            generated_images = generator.predict(noise)
            real_images = x_train[np.random.randint(0, x_train.shape[0], batch_size)]

            discriminator_loss_real = discriminator.train_on_batch(real_images, np.ones((batch_size, 1)))
            discriminator_loss_generated = discriminator.train_on_batch(generated_images, np.zeros((batch_size, 1)))

            discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_generated)

            noise = tf.random.normal((batch_size, 100))
            gan_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

        print(f"Epoch {epoch + 1}/{epochs}, Discriminator Loss: {discriminator_loss}, GAN Loss: {gan_loss}")

        if (epoch + 1) % 100 == 0:
            save_generated_image(epoch + 1, generator)

        if (epoch + 1) % 1000 == 0:
            save_models(epoch + 1, generator, discriminator)

# Função para salvar imagens geradas
def save_generated_image(epoch, generator, examples=10, dim=(1, 10), figsize=(10, 1)):
    noise = tf.random.normal((examples, 100))
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5
    plt.figure(figsize=figsize)
    for i in range(examples):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"gan_generated_images/gan_generated_image_epoch_{epoch}.png")
    plt.close()

# Função para salvar modelos da GAN
def save_models(epoch, generator, discriminator):
    generator.save(f"gan_models/generator_model_epoch_{epoch}.h5")
    discriminator.save(f"gan_models/discriminator_model_epoch_{epoch}.h5")

# Treinar a GAN
train_gan(epochs=10000, batch_size=32)
