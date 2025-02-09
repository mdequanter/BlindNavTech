void setup() {
    Serial.begin(115200); // Start serial communication at 115200 baud
    pinMode(9, INPUT_PULLUP); // Set pin 8 as input with internal pull-up resistor
}

void loop() {
    int xValue = analogRead(A0); // Read X-axis value
    int yValue = analogRead(A1); // Read Y-axis value
    int buttonState = digitalRead(9); // Read button state (0 = pressed, 1 = not pressed)

    Serial.print("X: ");
    Serial.print(xValue);
    Serial.print(" Y: ");
    Serial.print(yValue);
    Serial.print(" Button: ");
    Serial.println(buttonState);

    // No delay, so the loop runs as fast as possible
}
