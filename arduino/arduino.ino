#include <NewPing.h>//baixe a biblioteca direto da IDE

#define TRIG_PIN 9 // Pino do TRIG
#define ECHO_PIN 10 // Pino do ECHO
#define MAX_DISTANCE 200 // Distância máxima em centímetros

NewPing sonar(TRIG_PIN, ECHO_PIN, MAX_DISTANCE);
int ledPin = 13; // Pino do LED

void setup() {
  pinMode(ledPin, OUTPUT);
  Serial.begin(9600);
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();
    if (command == '1') {
      digitalWrite(ledPin, HIGH); // Liga o LED
    } else if (command == '0') {
      digitalWrite(ledPin, LOW); // Desliga o LED
    }
  }

  // Medir a distância com o sensor ultrassônico
  unsigned int distancia = sonar.ping_cm();
  Serial.println(distancia);
  delay(250); // Regular tempo de delay para calcular distancia (quantidades gigantes de GET)
}
