#!/usr/bin/env python3
"""
Simple PIR sensor test script.
Run on Pi to verify sensor is connected and working.
"""

import time
import sys

GPIO_PIN = 17  # Default pin, change if needed

try:
    import RPi.GPIO as GPIO
except ImportError:
    print("ERROR: RPi.GPIO not available. Are you running on a Raspberry Pi?")
    sys.exit(1)

print(f"Testing PIR sensor on GPIO {GPIO_PIN}")
print("Wave your hand in front of the sensor...")
print("Press Ctrl+C to stop\n")

GPIO.setmode(GPIO.BCM)
GPIO.setup(GPIO_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

try:
    motion_count = 0
    while True:
        state = GPIO.input(GPIO_PIN)
        if state:
            motion_count += 1
            print(f"[{motion_count}] MOTION DETECTED! (pin HIGH)")
            time.sleep(1)  # Debounce
        else:
            print(f"Waiting... (pin LOW)", end="\r")
        time.sleep(0.1)
except KeyboardInterrupt:
    print(f"\n\nTest complete. Detected {motion_count} motion events.")
finally:
    GPIO.cleanup(GPIO_PIN)
