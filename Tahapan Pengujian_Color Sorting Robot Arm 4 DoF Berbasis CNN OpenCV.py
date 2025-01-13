import cv2
import numpy as np
from keras.models import load_model
import RPi.GPIO as GPIO
import time

# Memuat model yang telah dilatih
model = load_model("best_model.keras")

# Label untuk kelas objek
label_to_index = {0: 'biru', 1: 'hitam', 2: 'kuning', 3: 'merah', 4: 'null'}

# Konfigurasi GPIO untuk servo
GPIO.setmode(GPIO.BCM)
servo_pins = {
    'servo1': 17,  # Pin GPIO untuk servo 1
    'servo2': 27,  # Pin GPIO untuk servo 2
    'servo3': 22,  # Pin GPIO untuk servo 3
    'servo4': 23   # Pin GPIO untuk servo 4
}

# Mengatur pin GPIO sebagai output
for pin in servo_pins.values():
    GPIO.setup(pin, GPIO.OUT)

# Inisialisasi PWM untuk servo (frekuensi 50Hz)
servo_pwm = {label: GPIO.PWM(pin, 50) for label, pin in servo_pins.items()}
for pwm in servo_pwm.values():
    pwm.start(0)  # Memulai PWM dengan duty cycle 0

def set_servo_angle(servo_label, angle):
    """Mengatur servo tertentu ke sudut tertentu."""
    if servo_label in servo_pwm:
        duty_cycle = (angle / 18.0) + 2.5  # Menghitung duty cycle dari sudut
        pwm = servo_pwm[servo_label]
        pwm.ChangeDutyCycle(duty_cycle)
        time.sleep(0.5)
        pwm.ChangeDutyCycle(0)  # Matikan sinyal PWM

def move_servo_based_on_color(color):
    """Menggerakkan servo berdasarkan warna yang terdeteksi."""
    if color == "null":
        set_servo_angle('servo1', 90)
        set_servo_angle('servo2', 90)
        set_servo_angle('servo3', 90)
        set_servo_angle('servo4', 90)
    elif color == "biru":
        set_servo_angle('servo1', 90)
        set_servo_angle('servo2', 155)
        set_servo_angle('servo3', 125)
        set_servo_angle('servo4', 135)
        set_servo_angle('servo2', 80)
        set_servo_angle('servo1', 45)
        set_servo_angle('servo2', 140)
        set_servo_angle('servo3', 60)
        set_servo_angle('servo4', 45)
    elif color == "kuning":
        set_servo_angle('servo1', 90)
        set_servo_angle('servo2', 155)
        set_servo_angle('servo3', 125)
        set_servo_angle('servo4', 135)
        set_servo_angle('servo2', 80)
        set_servo_angle('servo1', 143)
        set_servo_angle('servo2', 140)
        set_servo_angle('servo3', 60)
        set_servo_angle('servo4', 45)
    elif color == "merah":
        set_servo_angle('servo1', 90)
        set_servo_angle('servo2', 155)
        set_servo_angle('servo3', 125)
        set_servo_angle('servo4', 135)
        set_servo_angle('servo2', 80)
        set_servo_angle('servo1', 10)
        set_servo_angle('servo2', 140)
        set_servo_angle('servo3', 60)
        set_servo_angle('servo4', 45)
    elif color == "hitam":
        set_servo_angle('servo1', 90)
        set_servo_angle('servo2', 155)
        set_servo_angle('servo3', 125)
        set_servo_angle('servo4', 135)
        set_servo_angle('servo2', 80)
        set_servo_angle('servo1', 180)
        set_servo_angle('servo2', 140)
        set_servo_angle('servo3', 60)
        set_servo_angle('servo4', 45)

def classify_webcam():
    """Fungsi utama untuk mendeteksi warna menggunakan webcam."""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    if not cap.isOpened():
        print("Webcam tidak dapat diakses!")
        return

    previous_label = None  # Untuk menghindari gerakan servo berulang

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Gagal membaca frame dari webcam!")
                break

            resized_frame = cv2.resize(frame, (100, 100))
            normalized_frame = resized_frame / 255.0
            input_data = np.expand_dims(normalized_frame, axis=0)

            predictions = model.predict(input_data, verbose=0)  # Suppress output
            predicted_class = np.argmax(predictions)
            label = label_to_index[predicted_class]

            if label != previous_label:
                move_servo_based_on_color(label)
                previous_label = label

            cv2.putText(frame, f"Predicted: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow('Webcam', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

try:
    classify_webcam()
finally:
    for pwm in servo_pwm.values():
        pwm.stop()
    GPIO.cleanup()
