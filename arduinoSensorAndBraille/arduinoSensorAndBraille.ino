#include <Servo.h>

#define TRIG_PIN 2   // Ultrasone sensor Trigger Pin
#define ECHO_PIN 3   // Ultrasone sensor Echo Pin
#define BUTTON_JOYSTICK 9
const int SERVO_PIN = 6;
int currentPos = 0;
Servo servo;
int pulseWidth = 1500;
int lastPulse = 1500;
int delaySwitch = 100;
int delayBasic = 500;
bool objectAvoidance = true;
bool objectLocate = false;


int readDistance(){
  int distance;
  digitalWrite(TRIG_PIN, LOW); // Set the trigger pin to low for 2uS
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH); // Send a 10uS high to trigger ranging
  delayMicroseconds(20);
  digitalWrite(TRIG_PIN, LOW); // Send pin low again
  distance = pulseIn(ECHO_PIN, HIGH)/58; // Read in times pulse
  //delay(50);// Wait 50mS before next ranging
  return distance;
}


void setup() {
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(BUTTON_JOYSTICK, INPUT_PULLUP); // Set pin 8 as input with internal pull-up resistor

  servo.attach(SERVO_PIN);
  currentPos = servo.readMicroseconds();
  Serial.begin(9600);
  Serial.println("Avoidance started");
  delay(2000); // Wacht op de seriële initialisatie
}

void loop() {
  float distance;


  if (objectAvoidance == true) {
    distance = readDistance();
    if (distance <= 150 and distance > 10 ) {
      //Serial.println(distance);
 
      if (distance <= 150 and distance > 100 ) {
        delaySwitch = 180;
        pulseWidth = 984; // 3
      }
      if (distance <= 100 and distance > 50 ) {
        delaySwitch = 150;
        pulseWidth = 2024; // 2
      }
      if (distance <= 50 and distance > 30 ) {
        delaySwitch = 80;
        pulseWidth = 1864; // 1
      }
      if (distance <= 30 and distance > 10 ) {
        delaySwitch = 50;
        pulseWidth = 1574; // 0
      }


      if (lastPulse == pulseWidth) {
        servo.writeMicroseconds(pulseWidth-100);
        delay(100);
      }

      servo.writeMicroseconds(pulseWidth);
      delay(delaySwitch);
      lastPulse = pulseWidth;

    }
  }


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
        case 'M':
          currentPos = servo.readMicroseconds();
          pulseWidth = currentPos + 10;
          break;
        case 'P':
          currentPos = servo.readMicroseconds();
          pulseWidth = currentPos - 10;
          break;
        case 'O': // Obstackle 
          if (delaySwitch == 500 ) {
            delaySwitch = 100;
            objectAvoidance = true;
          } else {
            delaySwitch = 500;
            objectAvoidance = false;
          }
          pulseWidth = 1274;
          break;
        case 'L': // Obstackle 
          if (objectLocate == false ) {
            delaySwitch = 100;
            objectLocate = true;
          } else {
            delaySwitch = 500;
            objectLocate = false;
          }
          pulseWidth = 1274;
          break;
        default:
          pulseWidth = 1500; // Neutrale positie
          break;
      }
      
      if (lastPulse == pulseWidth) {
        servo.writeMicroseconds(pulseWidth-100);
        delay(100);
      }
      lastPulse = pulseWidth;
      /*
      Serial.print(pulseWidth);
      Serial.print(":");
      Serial.println(delaySwitch);
      */
      servo.writeMicroseconds(pulseWidth);
      delay(delaySwitch);
    }
  }

    int buttonState = digitalRead(BUTTON_JOYSTICK); // Read button state (0 = pressed, 1 = not pressed)
    if (objectLocate == true) {
      int yValue = analogRead(A0); // Read X-axis value
      int xValue = analogRead(A1); // Read Y-axis value
      // Print JSON output
      Serial.print("{\"X\":");
      Serial.print(xValue);
      Serial.print(",\"Y\":");
      Serial.print(yValue);
      Serial.print(",\"Button\":");
      Serial.print(buttonState);
      Serial.println("}");
      delay(200);
    }



}