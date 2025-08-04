import RPi.GPIO as GPIO
import time

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(16, GPIO.OUT)

servo_motor = GPIO.PWM(16,50)
servo_motor.start(0)

def unlock_door():
    """
    Unlocking the door using a servo motor.
    """
    is_locked = False
    print('Waiting 2 Seconds!')
    time.sleep(2)

    print("Turning Back 90 Degrees")
    servo_motor.ChangeDutyCycle(10.5)
    time.sleep(0.3)
    servo_motor.ChangeDutyCycle(0)
    print("Unlock button clicked.")

def lock_door():
    """
    Placeholder function for locking the door using a servo motor.
    """
    is_locked = True
    print("Turning back to 0 degrees")
    servo_motor.ChangeDutyCycle(7.5)
    print("Waiting 0.5 seconds!")
    time.sleep(0.3)
    servo_motor.ChangeDutyCycle(0)
    time.sleep(0.7)

    print("Lock button clicked.")

while True:
    user_input = input('Enter either u or l!')
    if user_input == 'u':
        unlock_door()
    elif user_input == 'l':
        lock_door()
    elif user_input.lower() == 'stop':
        servo_motor.stop()
        GPIO.cleanup()
        print("All Done! Goodbye!")
        break
    else:
        print('Invalid Input')
    
servo_motor.stop()
GPIO.cleanup()