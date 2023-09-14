import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Flatten, Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import random
import matplotlib.pyplot as plt

# Configuração do NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Exemplo de conjunto de dados condicionado por texto (texto e imagem correspondente)
dataset = [
    ("um gato preto", np.array([0.1, 0.2, 0.3])),
    ("um cachorro marrom", np.array([0.7, 0.4, 0.2])),
    ("uma flor vermelha", np.array([0.9, 0.5, 0.1]))
]

# Pré-processamento de texto
stop_words = set(stopwords.words('portuguese'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
    return " ".join(tokens)

# Crie um dicionário de palavras para índices
word_to_index = {}
index_to_word = {}
index = 1

for text, _ in dataset:
    processed_text = preprocess_text(text)
    for word in processed_text.split():
        if word not in word_to_index:
            word_to_index[word] = index
            index_to_word[index] = word
            index += 1

vocab_size = len(word_to_index) + 1  # Tamanho do vocabulário

# Modelo GAN condicionado por texto
def build_generator():
    model = Sequential()
    model.add(Dense(256, input_dim=vocab_size))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='sigmoid'))
    return model

def build_discriminator():
    model = Sequential()
    model.add(Dense(64, input_dim=3))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Compilar o discriminador e o modelo GAN
generator = build_generator()
discriminator = build_discriminator()

discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
discriminator.trainable = False

gan_input = tf.keras.layers.Input(shape=(vocab_size,))
x = generator(gan_input)
gan_output = discriminator(x)

gan = tf.keras.models.Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# Função para salvar uma imagem gerada
def save_generated_image(epoch, generator, examples=1, filename="gan_generated_image.png"):
    noise = tf.random.normal((examples, vocab_size))
    generated_images = generator.predict(noise)
    generated_images = (generated_images + 1) / 2.0  # Ajuste para o intervalo [0, 1]

    # Certifique-se de que generated_images tenha a forma correta
    generated_images = generated_images.reshape((examples, 128, 128, 3))

    plt.figure(figsize=(6, 6))
    plt.imshow(generated_images[0])
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# Treinamento da GAN
def train_gan(epochs, batch_size, save_interval):
    for epoch in range(epochs):
        for _ in range(len(dataset) // batch_size):
            batch_texts = []
            batch_images = []

            for _ in range(batch_size):
                text, image = random.choice(dataset)
                batch_texts.append(text)
                batch_images.append(image)

            batch_texts = [preprocess_text(text) for text in batch_texts]
            batch_indices = [[word_to_index[word] for word in text.split()] for text in batch_texts]

            noise = np.zeros((batch_size, vocab_size))
            for i, indices in enumerate(batch_indices):
                noise[i, indices] = 1

            generated_images = generator.predict(noise)
            real_images = (np.array(batch_images) * 2) - 1  # Ajuste das imagens reais para o intervalo [-1, 1]

            discriminator_loss_real = discriminator.train_on_batch(real_images, np.ones((batch_size, 1)))
            discriminator_loss_generated = discriminator.train_on_batch(generated_images, np.zeros((batch_size, 1)))

            discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_generated)

            noise = np.zeros((batch_size, vocab_size))
            for i, indices in enumerate(batch_indices):
                noise[i, indices] = 1
                gan_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
                print(f"Epoch {epoch + 1}/{epochs}, Discriminator Loss: {discriminator_loss}, GAN Loss: {gan_loss}")
                if (epoch + 1) % save_interval == 0:
                    save_generated_image(epoch + 1, generator)

# Treinar a GAN
train_gan(epochs=10000, batch_size=3, save_interval=100)
