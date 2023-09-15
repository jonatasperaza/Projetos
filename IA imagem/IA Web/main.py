import os
import webbrowser
import subprocess

# Função para ler a base de dados e encontrar imagens correspondentes
def read_database_and_display(query, database_dir):
    # Ler o arquivo de consultas
    queries_file_path = os.path.join(database_dir, 'queries.txt')
    if not os.path.exists(queries_file_path):
        print("A base de dados está vazia.")
        return

    with open(queries_file_path, 'r') as queries_file:
        queries = queries_file.read().splitlines()

    # Verificar se a consulta fornecida está na base de dados
    if query in queries:
        # Encontrar a imagem correspondente e abri-la no navegador
        img_filename = os.path.join(database_dir, f"{query}.jpg")
        if os.path.exists(img_filename):
            webbrowser.open(img_filename)
        else:
            print("Imagem correspondente não encontrada.")
    else:
        print(f"Consulta '{query}' não encontrada na base de dados. Executando pesquisa...")

        # Executar o primeiro código para pesquisar e salvar a imagem
        subprocess.call(['python', 'basedados.py', query, database_dir])

# Pedir ao usuário uma consulta para buscar na base de dados
search_query = input("Digite uma consulta para buscar na base de dados: ")
database_directory = "database"  # Diretório onde a base de dados está localizada

# Chamar a função para ler a base de dados e exibir a imagem correspondente
read_database_and_display(search_query, database_directory)
