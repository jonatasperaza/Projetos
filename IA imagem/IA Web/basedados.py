import requests
from bs4 import BeautifulSoup
import webbrowser
import os

# Função para fazer a pesquisa e salvar a imagem e consulta na base de dados
def save_to_database(query, database_dir):
    # Formatar a consulta para uma pesquisa no Google Images
    query = query.replace(' ', '+')
    url = f"https://www.google.com.br/search?q={query}&tbm=isch"

    # Fazer a solicitação HTTP e obter o conteúdo da página
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Encontrar a URL da primeira imagem na página
    img_tags = soup.find_all('img')
    if img_tags:
        # A primeira imagem real geralmente está no índice 1 (índice 0 é o logotipo do Google)
        img_url = img_tags[1].get('src')

        # Salvar a imagem em um diretório da base de dados
        img_filename = os.path.join(database_dir, f"{query}.jpg")
        with open(img_filename, 'wb') as img_file:
            img_file.write(requests.get(img_url).content)

        # Salvar a consulta na base de dados
        with open(os.path.join(database_dir, 'queries.txt'), 'a') as queries_file:
            queries_file.write(query + '\n')

        print(f"Consulta '{query}' e imagem correspondente salvas na base de dados.")
    else:
        print(f"Nenhuma imagem encontrada para a consulta: {query}")

# Pedir ao usuário um nome para pesquisar e salvar na base de dados
search_query = input("Digite um nome para pesquisar e salvar na base de dados: ")
database_directory = "database"  # Diretório onde a base de dados será criada

# Certifique-se de que o diretório da base de dados exista
if not os.path.exists(database_directory):
    os.makedirs(database_directory)

# Chamar a função para salvar na base de dados
save_to_database(search_query, database_directory)
