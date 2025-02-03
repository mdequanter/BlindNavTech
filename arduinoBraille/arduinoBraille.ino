#include <Servo.h>

const int SERVO_PIN = 6;
int currentPos = 0;
Servo servo;
int pulseWidth = 1500;
int lastPulse = 1500;
int delaySwitch = 500;
int delayBasic = 500;


void setup() {
  servo.attach(SERVO_PIN);
  currentPos = servo.readMicroseconds();
  Serial.begin(9600);
  Serial.println(currentPos);
  delay(2000); // Wacht op de seriële initialisatie
}

void loop() {
  // Controleer of er seriële data beschikbaar is
  if (Serial.available() > 0) {
    char c = Serial.read(); // Lees een karakter
    if (c != '\n' && c != '\r') {

      switch (c) {
        case '0':
          pulseWidth = 1574; // Minimale servo positie
          break;
        case '1':
          pulseWidth = 1864;
          break;
        case '2':
          pulseWidth = 2024;
          break;
        case '3':
          pulseWidth = 984;
          break;
        case '4':
          pulseWidth = 1354;
          break;
        case '5':
          pulseWidth = 1784;
          break;
        case '6':
          pulseWidth = 1474;
          break;
        case '7':
          pulseWidth = 1274;
          break;
        case '8':
          pulseWidth = 1164;
          break;
        case '9':
          pulseWidth = 1094;
          break;
        case ' ':
          pulseWidth = 2224;
          break;
        case ':':
          pulseWidth = 1700;
          break;
        case '-':
          pulseWidth = 1684;
          break;
        case 'L':
          currentPos = servo.readMicroseconds();
          pulseWidth = currentPos + 10;
          break;
        case 'R':
          currentPos = servo.readMicroseconds();
          pulseWidth = currentPos - 10;
          break;
        case 'O': // Obstackle 
          if (delaySwitch == 500 ) {
            delaySwitch = 100;
          } else {
            delaySwitch = 500;
          }
          pulseWidth = 1274;
        default:
          pulseWidth = 1500; // Neutrale positie
          break;
      }
      
      if (lastPulse == pulseWidth) {
        servo.writeMicroseconds(pulseWidth-100);
        delay(100);
      }
      
      lastPulse = pulseWidth;
      Serial.print(pulseWidth);
      Serial.print(":");
      Serial.println(delaySwitch);
      servo.writeMicroseconds(pulseWidth);
      delay(delaySwitch);
    }
  }
}