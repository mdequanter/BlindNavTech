#!/usr/bin/python3
# encoding: utf-8

import RPi.GPIO as GPIO
import time
import numpy as np
import pyaudio

# Set GPIO mode
GPIO.setmode(GPIO.BCM)

# Define GPIO pins
TRIG = 15  # TRIG pin
ECHO = 14  # ECHO pin

# Set up GPIO pins
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

# Initialize PyAudio for sound output
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=44100,
                output=True)

print("Distance measurement with sound feedback (JSN-SR04T)")

def generate_click(frequency=1000, duration=0.02):
    """Generate a short click sound"""
    sample_rate = 44100  # Sample rate in Hz
    samples = (np.sin(2 * np.pi * np.arange(sample_rate * duration) * frequency / sample_rate)).astype(np.float32)
    sound = (samples * 32767).astype(np.int16).tobytes()
    stream.write(sound)

try:
    while True:
        # Ensure TRIG is LOW before triggering
        GPIO.output(TRIG, False)
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
                break  

        # Wait for ECHO to go LOW (end time)
        timeout = time.time() + 0.1  # Reset timeout
        while GPIO.input(ECHO) == 1:
            pulse_end = time.time()
            if time.time() > timeout:
                print("Timeout waiting for pulse end")
                break  

        # If pulse_start or pulse_end is None, skip distance calculation
        if pulse_start is None or pulse_end is None:
            print("Measurement failed, retrying...")
            continue  

        # Calculate distance
        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 17150  # Convert time to distance in cm
        distance = round(distance, 2)  # Round to two decimal places

        print (distance)

        MIN_DISTANCE = 25     # cm (very close)
        MAX_DISTANCE = 100    # cm (1 meters)
        MIN_INTERVAL = 0.1    # seconds (fast clicking)
        MAX_INTERVAL = 0.3    # seconds (slow clicking)


        if MIN_DISTANCE < distance < MAX_DISTANCE:  
            # print(f"Distance: {distance - 0.5} cm")  # Calibration correction
            
            # Define the min and max distances for interpolation

            # Ensure the distance is within the defined range
            if distance < MIN_DISTANCE:
                distance = MIN_DISTANCE
            elif distance > MAX_DISTANCE:
                distance = MAX_DISTANCE

            # Linear interpolation formula
            click_interval = MIN_INTERVAL + (MAX_INTERVAL - MIN_INTERVAL) * (distance - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE)

            # Play sound with calculated interval
            generate_click(frequency=1000)  # Play beep
            time.sleep(click_interval)  # Control click speed

except KeyboardInterrupt:
    print("\nMeasurement stopped by user")
    GPIO.cleanup()
    stream.stop_stream()
    stream.close()
    p.terminate()
