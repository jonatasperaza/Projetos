import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image

# Função para gerar uma imagem com base em um modelo GAN simples
def generate_image(generator):
    noise = np.random.normal(0, 1, (1, 100))  # Gere ruído aleatório
    generated_image = generator.predict(noise)[0]
    generated_image = (generated_image * 255).astype(np.uint8)
    img = Image.fromarray(generated_image)
    return img

# Função para treinar um modelo GAN simples
def train_simple_gan(epochs=10000):
    generator = models.Sequential()
    generator.add(layers.Dense(128, input_dim=100, activation='relu'))
    generator.add(layers.Dense(3 * 800 * 600, activation='sigmoid'))  # Agora geramos 3 canais (RGB)
    generator.add(layers.Reshape((800, 600, 3)))  # Redimensionamos para 3 canais e 800x600

    discriminator = models.Sequential()
    discriminator.add(layers.Flatten(input_shape=(800, 600, 3)))
    discriminator.add(layers.Dense(128, activation='relu'))
    discriminator.add(layers.Dense(1, activation='sigmoid'))

    discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    gan_input = layers.Input(shape=(100,))
    x = generator(gan_input)
    gan_output = discriminator(x)

    gan = models.Model(gan_input, gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')

    for epoch in range(epochs):
        noise = np.random.normal(0, 1, (128, 100))  # Gere um lote de ruído
        generated_images = generator.predict(noise)

        real_images = np.random.randn(128, 800, 600, 3)  # Gere imagens reais aleatórias (pode ser substituído pelo seu banco de dados)

        x = np.concatenate((real_images, generated_images), axis=0)  # Concatene ao longo do eixo 0 (lote)
        y = np.zeros(2 * 128)
        y[:128] = 1  # Rótulos para imagens reais

        discriminator_loss = discriminator.train_on_batch(x, y)

        noise = np.random.normal(0, 1, (128, 100))  # Gere um novo lote de ruído
        y_gan = np.ones(128)  # Rótulos para imagens geradas (queremos que o gerador acredite que são reais)

        gan_loss = gan.train_on_batch(noise, y_gan)

        print(f'Epoch {epoch}/{epochs}, Discriminator Loss: {discriminator_loss[0]}, GAN Loss: {gan_loss}')

        if epoch % 100 == 0:
            # Salva uma imagem gerada a cada 100 épocas
            generated_image = generate_image(generator)
            generated_image.save(f"generated_image_epoch_{epoch}.jpg")

    # Salva o modelo GAN treinado
    generator.save('gan_generator_model.h5')
    discriminator.save('gan_discriminator_model.h5')
    gan.save('gan_model.h5')

# Treina o modelo GAN simples e salva os modelos treinados
train_simple_gan()
