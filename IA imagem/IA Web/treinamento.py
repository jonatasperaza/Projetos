import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image
import os
import cv2

# Limitar o uso de memória RAM do TensorFlow
ram_memory_limit = 10240  # Defina o limite máximo de memória RAM (em MB)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Configuração para evitar alocação dinâmica de memória GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # Limite a alocação de memória RAM
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],  # Use a primeira GPU (se disponível)
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=ram_memory_limit)]
        )

    except RuntimeError as e:
        print(e)

# Função para gerar uma imagem com base em um modelo GAN simples
def generate_image(generator):
    noise = np.random.normal(0, 1, (1, 100))  # Gere ruído aleatório
    generated_image = generator.predict(noise)[0]
    generated_image = (generated_image * 255).astype(np.uint8)
    img = Image.fromarray(generated_image)
    img = img.convert('RGB')  # Converta a imagem para o modo RGB
    return img

# Função para carregar imagens reais de uma pasta
def load_real_images(dataset_path):
    image_list = []
    for filename in os.listdir(dataset_path):
        if filename.endswith(".jpg"):  # Verifique se é um arquivo de imagem
            img = cv2.imread(os.path.join(dataset_path, filename))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converta de BGR para RGB
                img = cv2.resize(img, (800, 600))  # Redimensione para 800x600
                image_list.append(img)
    return np.array(image_list)

# Função para treinar um modelo GAN simples
def train_simple_gan(epochs=100):
    generator = models.Sequential()
    generator.add(layers.Dense(128, input_dim=100, activation='relu'))
    generator.add(layers.Dense(3 * 800 * 600, activation='tanh'))  # Agora geramos 3 canais (RGB) com 'tanh'
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

    dataset_path = 'database1/'  # Substitua pelo caminho real da sua pasta de imagens reais
    real_images = load_real_images(dataset_path)

    for epoch in range(epochs):
        noise = np.random.normal(0, 1, (128, 100))  # Gere um lote de ruído
        generated_images = generator.predict(noise)

        x = np.concatenate((real_images, generated_images), axis=0)  # Concatene ao longo do eixo 0 (lote)
        y = np.zeros(2 * 128)
        y[:128] = 1  # Rótulos para imagens reais

        noise = np.random.normal(0, 1, (128, 100))  # Gere um novo lote de ruído
        y_gan = np.ones(128)  # Rótulos para imagens geradas (queremos que o gerador acredite que são reais)

        print(f'Epoch {epoch}/{epochs}')

        if epoch % 10 == 0:
            # Salva uma imagem gerada a cada 10 épocas  
            generated_image = generate_image(generator)
            generated_image.save(f"generated_image_epoch_{epoch}.jpg")

    # Salva o modelo GAN treinado
    generator.save('gan_generator_model.h5')
    discriminator.save('gan_discriminator_model.h5')
    gan.save('gan_model.h5')

# Treina o modelo GAN simples e salva os modelos treinados
train_simple_gan()
