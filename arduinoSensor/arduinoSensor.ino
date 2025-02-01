#include <Servo.h>

#define TRIG_PIN 2   // Ultrasone sensor Trigger Pin
#define ECHO_PIN 3   // Ultrasone sensor Echo Pin
#define BUZZER_PIN 4 // Buzzer Pin
#define SERVO_PIN 5  // Servo Pin

Servo myServo; // Maak een servo object
int vorigeWaarde = -1; // Houd de vorige servo-positie bij
int nieuweWaarde = 90;
void setup() {
    pinMode(TRIG_PIN, OUTPUT);
    pinMode(ECHO_PIN, INPUT);
    pinMode(BUZZER_PIN, OUTPUT);
    Serial.begin(115200);  // Pas aan naar 9600 als nodig
    
    myServo.attach(SERVO_PIN); // Koppel de servo aan de pin
    myServo.write(0); // Zet de servo op 0° als startwaarde
}

void loop() {
    long duration;
    float distance;

    // Stuur triggerpuls
    digitalWrite(TRIG_PIN, LOW);
    delayMicroseconds(2);
    digitalWrite(TRIG_PIN, HIGH);
    delayMicroseconds(10);
    digitalWrite(TRIG_PIN, LOW);

    // Meet echo duur
    duration = pulseIn(ECHO_PIN, HIGH,20000);
    distance = (duration * 0.0343) / 2;  // Omrekenen naar cm
    //Serial.println(distance);

  if (distance <= 200 and distance > 100 ) {
        myServo.write(90); // Beweeg de servo
        delay(180);
        myServo.write(100);
        delay(180);
        myServo.write(90);
    }

    if (distance <= 100 and distance > 50 ) {
        myServo.write(90); // Beweeg de servo
        delay(150);
        myServo.write(100);
        delay(150);
        myServo.write(90);
    }
    if (distance <= 50 and distance > 30) {
        myServo.write(90); // Beweeg de servo
        delay(80);
        myServo.write(100);
        delay(80);
        myServo.write(90);
    }

    if (distance <= 30 and distance >= 10) {
        myServo.write(90); // Beweeg de servo
        delay(50);
        myServo.write(100);
        delay(50);
        myServo.write(90);
    }
 
    delay(10);  // Pas snelheid aan op basis van afstand
}
