import cv2
import numpy as np
import dlib
import pigpio
import time
from simple_pid import PID

# Initialize Dlib Face Detector
cnn_face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

# Initialize Camera
cap = cv2.VideoCapture(0)
frame_width, frame_height = 640, 480
cap.set(3, frame_width)
cap.set(4, frame_height)

# Initialize Raspberry Pi Servo Control (GPIO 17 for Pan, GPIO 18 for Tilt)
pan_pin, tilt_pin = 17, 18
pi = pigpio.pi()
pi.set_servo_pulsewidth(pan_pin, 1500)
pi.set_servo_pulsewidth(tilt_pin, 1500)

# PID Controllers for Smooth Movement
pan_pid = PID(0.1, 0.01, 0.05, setpoint=frame_width // 2)
tilt_pid = PID(0.1, 0.01, 0.05, setpoint=frame_height // 2)
pan_pid.output_limits = (-200, 200)
tilt_pid.output_limits = (-200, 200)

# Servo Angle Ranges
servo_min, servo_max = 500, 2500
pan_angle, tilt_angle = 1500, 1500

def update_servo(pin, angle):
    angle = max(servo_min, min(servo_max, angle))
    pi.set_servo_pulsewidth(pin, angle)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to RGB for Dlib
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = cnn_face_detector(rgb_frame, 1)
    
    if faces:
        for face in faces:
            x, y, w, h = face.rect.left(), face.rect.top(), face.rect.width(), face.rect.height()
            cx, cy = x + w // 2, y + h // 2
            
            # PID Control Updates
            pan_adjust = pan_pid(cx)
            tilt_adjust = tilt_pid(cy)
            
            pan_angle += pan_adjust
            tilt_angle -= tilt_adjust  # Inverted tilt direction
            
            update_servo(pan_pin, pan_angle)
            update_servo(tilt_pin, tilt_angle)
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
    
    cv2.imshow('Face Tracking', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
pi.set_servo_pulsewidth(pan_pin, 0)
pi.set_servo_pulsewidth(tilt_pin, 0)
pi.stop()
