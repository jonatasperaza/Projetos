from flask import Flask, request, render_template
import serial

app = Flask(__name__)
arduino = serial.Serial('COM6', 9600)  # Ajuste a porta COM conforme necessário

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ligar')
def ligar_led():
    arduino.write(b'1\n')  # Envia '1' para ligar o LED
    return 'LED ligado'

@app.route('/desligar')
def desligar_led():
    arduino.write(b'0\n')  # Envia '0' para desligar o LED
    return 'LED desligado'

@app.route('/obter_distancia')  # Defina a rota para obter a distância
def obter_distancia():
    # Aqui você pode adicionar código para obter a distância do sensor ultrassônico
    # Suponha que você já tenha lógica para obter a distância e armazená-la em uma variável
    distancia = arduino.readline().decode().strip()  # Substitua pelo valor real da distância
    return str(distancia)  # Retorna a distância como uma string

if __name__ == '__main__':
    app.run(port=80)
