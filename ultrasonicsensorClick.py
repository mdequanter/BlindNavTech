import os
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

print("Distance measurement with sound feedback (JSN-SR04T)")

def generate_click():
    """Play a short click sound"""
    os.system("aplay click.wav")  # Plays a sound using system audio

try:
    while True:
        GPIO.output(TRIG, False)
        time.sleep(0.1)  # Short delay

        GPIO.output(TRIG, True)
        time.sleep(0.00001)
        GPIO.output(TRIG, False)

        pulse_start, pulse_end = None, None
        timeout = time.time() + 0.1

        while GPIO.input(ECHO) == 0:
            pulse_start = time.time()
            if time.time() > timeout:
                break  

        timeout = time.time() + 0.1
        while GPIO.input(ECHO) == 1:
            pulse_end = time.time()
            if time.time() > timeout:
                break  

        if pulse_start is None or pulse_end is None:
            continue  

        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 17150
        distance = round(distance, 2)

        MIN_DISTANCE = 25     # cm
        MAX_DISTANCE = 100    # cm
        MIN_INTERVAL = 0.1    # sec
        MAX_INTERVAL = 0.3    # sec

        if MIN_DISTANCE < distance < MAX_DISTANCE:
            click_interval = MIN_INTERVAL + (MAX_INTERVAL - MIN_INTERVAL) * (distance - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE)
            generate_click()
            time.sleep(click_interval)  # Adjusts click speed

except KeyboardInterrupt:
    print("\nMeasurement stopped by user")
    GPIO.cleanup()
