import openai
import os

def ler_chave_api():
    caminho_arquivo = os.path.join(os.path.dirname(__file__), "api_key.txt")
    try:
        with open(caminho_arquivo, "r") as file:
            return file.read().strip()
    except FileNotFoundError:
        return None

def pensar_como_chatgpt(pergunta):
    chave_api = ler_chave_api()
    if chave_api is None:
        return "Erro: chave de API não encontrada"

    openai.api_key = chave_api

    resposta = openai.Completion.create(
        engine="text-davinci-003",
        prompt=pergunta,
        max_tokens=2000,
        temperature=1
    )
    return resposta.choices[0].text.strip()

while True:
    pergunta = input("Faça uma pergunta: ")
    if pergunta.lower() == "sair":
        break
    resposta = pensar_como_chatgpt(pergunta)
    print(resposta)
