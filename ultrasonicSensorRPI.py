#!/usr/bin/python3
# encoding: utf-8

import RPi.GPIO as GPIO
import time

# Set GPIO mode
GPIO.setmode(GPIO.BCM)

# Define GPIO pins
TRIG = 15  # TRIG pin
ECHO = 14  # ECHO pin

# Set up GPIO pins
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

print("Distance measurement in progress (JSN-SR04T)")

try:
    while True:
        # Ensure TRIG is LOW before triggering
        GPIO.output(TRIG, False)
        # print("Waiting for sensor to settle...")
        time.sleep(0.1)  # Shorter delay for JSN-SR04T

        # Send a 10-microsecond pulse
        GPIO.output(TRIG, True)
        time.sleep(0.00001)
        GPIO.output(TRIG, False)

        # Initialize variables
        pulse_start = None
        pulse_end = None
        timeout = time.time() + 0.1  # 100ms timeout

        # Wait for ECHO to go HIGH (start time)
        while GPIO.input(ECHO) == 0:
            pulse_start = time.time()
            if time.time() > timeout:
                print("Timeout waiting for pulse start")
                break  # Prevent infinite loop

        # Wait for ECHO to go LOW (end time)
        timeout = time.time() + 0.1  # Reset timeout for the next phase
        while GPIO.input(ECHO) == 1:
            pulse_end = time.time()
            if time.time() > timeout:
                print("Timeout waiting for pulse end")
                break

        # If pulse_start or pulse_end is None, skip distance calculation
        if pulse_start is None or pulse_end is None:
            print("Measurement failed, retrying...")
            continue  # Skip to the next loop iteration

        # Calculate distance
        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 17150  # Convert time to distance
        distance = round(distance, 2)  # Round to two decimal places

        if 20 < distance < 600:  # JSN-SR04T range: ~20cm to 600cm
            print(f"Distance: {distance - 0.5} cm")  # Calibration correction
        else:
            print("Out Of Range")

        time.sleep(0.1)  # Short delay before next measurement

except KeyboardInterrupt:
    print("\nMeasurement stopped by user")
    GPIO.cleanup()
