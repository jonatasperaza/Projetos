import os
import h5py
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Carregar o conjunto de dados MNIST
(x_train, _), (_, _) = mnist.load_data()
x_train = x_train / 127.5 - 1.0  # Normalização para o intervalo [-1, 1]

# Tamanho do vocabulário
vocab_size = 100  # Escolha o tamanho do vocabulário que desejar

# Modelo GAN condicionado por texto
def build_generator():
    model = Sequential()
    model.add(Dense(256, input_dim=vocab_size))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(784, activation='tanh'))  # 28x28x1 (MNIST)
    model.add(Reshape((28, 28, 1)))  # Redimensionar para o tamanho da imagem MNIST
    return model

def build_discriminator():
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28, 1)))  # Flatten para imagem MNIST
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Compilar o discriminador e o modelo GAN
generator = build_generator()
discriminator = build_discriminator()

discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
discriminator.trainable = False

# Definir a variável GAN aqui para que ela seja acessível em todo o código
gan_input = tf.keras.layers.Input(shape=(vocab_size,))
x = generator(gan_input)
gan_output = discriminator(x)
gan = tf.keras.models.Model(gan_input, gan_output)

# Compilar o modelo GAN
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# Verifique se o arquivo 'gan_model.h5' está vazio
if not os.path.isfile('gan_model.h5') or os.path.getsize('gan_model.h5') == 0:
    print("Arquivo 'gan_model.h5' está vazio. Iniciando treinamento e salvando modelos...")

    # Crie um arquivo HDF5 vazio
    with h5py.File('gan_model.h5', 'w') as file:
        # Crie grupos para o gerador, discriminador e GAN
        generator_group = file.create_group('generator')
        discriminator_group = file.create_group('discriminator')
        gan_group = file.create_group('gan')

        # Datasets vazios para os pesos dos modelos
        generator_group.create_dataset('weights', data=[])
        discriminator_group.create_dataset('weights', data=[])
        gan_group.create_dataset('weights', data=[])

        # Datasets para informações adicionais (se necessário)
        generator_group.create_dataset('info', data='Informações do gerador')
        discriminator_group.create_dataset('info', data='Informações do discriminador')
        gan_group.create_dataset('info', data='Informações do GAN')

    # Função para salvar os pesos dos modelos no arquivo
    def save_model_weights(generator, discriminator, gan):
        with h5py.File('gan_model.h5', 'a') as file:
            generator_weights = generator.get_weights()
            discriminator_weights = discriminator.get_weights()
            gan_weights = gan.get_weights()
            
            file['generator/weights'].resize(len(generator_weights))
            file['discriminator/weights'].resize(len(discriminator_weights))
            file['gan/weights'].resize(len(gan_weights))
            
            file['generator/weights'][:] = generator_weights
            file['discriminator/weights'][:] = discriminator_weights
            file['gan/weights'][:] = gan_weights

    # Função para salvar uma imagem gerada
    def save_generated_image(epoch, generator, examples=1, filename="gan_generated_image.png"):
        noise = np.random.normal(0, 1, (examples, vocab_size))
        generated_images = generator.predict(noise)
        generated_images = (generated_images + 1) / 2.0  # Ajuste para o intervalo [0, 1]

        plt.figure(figsize=(6, 6))
        for i in range(examples):
            plt.subplot(examples, 1, i+1)
            plt.imshow(generated_images[i, :, :, 0], cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    # Treinamento da GAN
    def train_gan(epochs, batch_size, save_interval):
        for epoch in range(epochs):
            for _ in range(len(x_train) // batch_size):
                batch_images = x_train[np.random.randint(0, x_train.shape[0], batch_size)]
                noise = np.random.normal(0, 1, (batch_size, vocab_size))

                generated_images = generator.predict(noise)
                discriminator_loss_real = discriminator.train_on_batch(batch_images, np.ones((batch_size, 1)))
                discriminator_loss_generated = discriminator.train_on_batch(generated_images, np.zeros((batch_size, 1)))
                discriminator_loss = 0.5 * (discriminator_loss_real + discriminator_loss_generated)

                noise = np.random.normal(0, 1, (batch_size, vocab_size))
                gan_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

                print(f"Epoch {epoch + 1}/{epochs}, Discriminator Loss: {discriminator_loss}, GAN Loss: {gan_loss}")

            if (epoch + 1) % save_interval == 0:
                save_generated_image(epoch + 1, generator)
                
                # Salvar o modelo GAN e os pesos do gerador e discriminador
                generator.save('generator_model.h5')
                discriminator.save('discriminator_model.h5')
                gan.save('gan_model.h5')

    # Treinar a GAN
    train_gan(epochs=10000, batch_size=64, save_interval=1)

else:
    print("Arquivo 'gan_model.h5' não está vazio. Carregando modelos existentes...")

    # Carregue os modelos existentes a partir do arquivo HDF5
    with h5py.File('gan_model.h5', 'r') as file:
        generator_weights = file['generator/weights'][:]
        discriminator_weights = file['discriminator/weights'][:]
        gan_weights = file['gan/weights'][:]

    # Crie modelos com as dimensões corretas
    generator = build_generator()
    discriminator = build_discriminator()
    gan_input = tf.keras.layers.Input(shape=(vocab_size,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = tf.keras.models.Model(gan_input, gan_output)

    # Defina os pesos carregados nos modelos
    generator.set_weights(generator_weights)
    discriminator.set_weights(discriminator_weights)
    gan.set_weights(gan_weights)

    # Continuar treinando se necessário
    train_gan(epochs=10000, batch_size=64, save_interval=1)
    gan.save('gan_model.h5')

# Função para salvar uma imagem gerada
def save_generated_image(epoch, generator, examples=1, filename="gan_generated_image.png"):
    noise = np.random.normal(0, 1, (examples, vocab_size))
    generated_images = generator.predict(noise)
    generated_images = (generated_images + 1) / 2.0  # Ajuste para o intervalo [0, 1]

    plt.figure(figsize=(6, 6))
    for i in range(examples):
        plt.subplot(examples, 1, i+1)
        plt.imshow(generated_images[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Treinamento da GAN
def train_gan(epochs, batch_size, save_interval):
    for epoch in range(epochs):
        for _ in range(len(x_train) // batch_size):
            batch_images = x_train[np.random.randint(0, x_train.shape[0], batch_size)]
            noise = np.random.normal(0, 1, (batch_size, vocab_size))

            generated_images = generator.predict(noise)
            discriminator_loss_real = discriminator.train_on_batch(batch_images, np.ones((batch_size, 1)))
            discriminator_loss_generated = discriminator.train_on_batch(generated_images, np.zeros((batch_size, 1)))
            discriminator_loss = 0.5 * (discriminator_loss_real + discriminator_loss_generated)

            noise = np.random.normal(0, 1, (batch_size, vocab_size))
            gan_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

            print(f"Epoch {epoch + 1}/{epochs}, Discriminator Loss: {discriminator_loss}, GAN Loss: {gan_loss}")

        if (epoch + 1) % save_interval == 0:
            save_generated_image(epoch + 1, generator)
            
            # Salvar o modelo GAN e os pesos do gerador e discriminador
            generator.save('generator_model.h5')
            discriminator.save('discriminator_model.h5')
            gan.save('gan_model.h5')

# Treinar a GAN
train_gan(epochs=10000, batch_size=64, save_interval=1)
