from PIL import Image, ImageDraw
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Função para carregar a base de dados (imagens de referência)
def load_database(database_dir):
    images = {}
    for filename in os.listdir(database_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_name = os.path.splitext(filename)[0]
            img_path = os.path.join(database_dir, filename)
            try:
                img = Image.open(img_path)
                images[img_name] = img
            except Exception as e:
                print(f"Erro ao carregar imagem {filename}: {str(e)}")
    return images

# Função para treinar um modelo de geração de imagens (exemplo simplificado)
def train_image_generation_model(database_dir):
    model = models.Sequential()
    model.add(layers.Dense(64, input_dim=100))
    model.add(layers.Dense(784, activation='sigmoid'))
    model.add(layers.Reshape((28, 28, 1)))

    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

# Função para gerar uma imagem com base em uma consulta
def generate_image(query, image_database, model):
    if query in image_database:
        reference_image = image_database[query]

        # Criar uma nova imagem com base na imagem de referência
        generated_image = Image.new("RGB", reference_image.size)
        draw = ImageDraw.Draw(generated_image)

        # Aqui você pode adicionar lógica para desenhar ou modificar a nova imagem com base na imagem de referência
        # Neste exemplo, estamos apenas copiando a imagem de referência
        generated_image.paste(reference_image)

        # Salvar a imagem gerada
        generated_image.save(f"{query}_generated.jpg")
        print(f"Imagem gerada com base na consulta: {query}_generated.jpg")
    else:
        print(f"Consulta '{query}' não encontrada na base de dados.")

# Função principal para interagir com o usuário
def main():
    database_directory = "IA imagem/IA Web/database/"  # Diretório onde a base de dados está localizada
    image_database = load_database(database_directory)
    model = train_image_generation_model(database_directory)

    while True:
        search_query = input("Digite o nome da imagem que você deseja gerar (ou 'sair' para sair): ")
        if search_query.lower() == "sair":
            break
        else:
            generate_image(search_query, image_database, model)

if __name__ == "__main__":
    main()
