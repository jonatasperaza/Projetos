import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape
from tensorflow.keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt

# Tamanho do vetor de ruído
noise_dim = 100

# Crie o gerador
def build_generator():
    model = Sequential()
    model.add(Dense(256, input_dim=noise_dim))
    model.add(Dense(784, activation='sigmoid'))  # Camada de saída com ativação sigmóide
    model.add(Reshape((28, 28, 1)))
    return model

# Gere uma imagem a partir de um vetor de ruído
def generate_image(generator, noise):
    image = generator.predict(noise)
    return image

# Crie o gerador
generator = build_generator()

# Gere uma imagem de exemplo
noise = np.random.normal(0, 1, (1, noise_dim))
generated_image = generate_image(generator, noise)

# Exiba a imagem gerada
plt.imshow(generated_image[0, :, :, 0], cmap='gray')
plt.axis('off')
plt.show()
