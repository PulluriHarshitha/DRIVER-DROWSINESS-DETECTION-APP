from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import time
import dlib
import cv2
import pygame
import os


# ==============================
# Initialize pygame mixer once
# ==============================
pygame.mixer.init()
sleep_channel = pygame.mixer.Channel(0)
yawn_channel = pygame.mixer.Channel(1)

# ✅ Set up your MP3 paths (edit if needed)
SLEEP_SOUND_PATH = r"C:\Users\rkarj\OneDrive\Desktop\Driver\Real-Time-Drowsiness-Detection-System\warning_sleep.mp3"
YAWN_SOUND_PATH = r"C:\Users\rkarj\OneDrive\Desktop\Driver\Real-Time-Drowsiness-Detection-System\warning_yawn.mp3"

sleep_sound = pygame.mixer.Sound(SLEEP_SOUND_PATH)
yawn_sound = pygame.mixer.Sound(YAWN_SOUND_PATH)

# ==============================
# EAR and MAR calculations
# ==============================
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    return (leftEAR + rightEAR) / 2.0, leftEye, rightEye


def mouth_aspect_ratio(shape):
    A = dist.euclidean(shape[50], shape[58])
    B = dist.euclidean(shape[52], shape[56])
    C = dist.euclidean(shape[48], shape[54])
    return (A + B) / (2.0 * C)


# ==============================
# Constants
# ==============================
EYE_AR_THRESH = 0.27
EYE_AR_CONSEC_FRAMES = 48
MAR_THRESH = 0.6
YAWN_CONSEC_FRAMES = 30
COOLDOWN_TIME = 5  # seconds before alarm can re-trigger

COUNTER = 0
YAWN_COUNTER = 0
sleep_alarm_on = False
yawn_alarm_on = False

last_sleep_alert = 0
last_yawn_alert = 0

sleep_alert_start_time = None
yawn_alert_start_time = None

# ==============================
# Paths
# ==============================
MODEL_PATH = r"C:\Users\rkarj\OneDrive\Desktop\Driver\Real-Time-Drowsiness-Detection-System\shape_predictor_68_face_landmarks.dat"

print("-> Loading the predictor and detector...")
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
detector = cv2.CascadeClassifier(cascade_path)
predictor = dlib.shape_predictor(MODEL_PATH)

print("-> Starting Video Stream")
vs = VideoStream(src=0).start()
time.sleep(1.0)

# ==============================
# Main Loop
# ==============================
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5,
        minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        ear, leftEye, rightEye = final_ear(shape)
        mar = mouth_aspect_ratio(shape)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        lip = shape[48:60]

        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

        current_time = time.time()

        # ==============================
        # Drowsiness Detection
        # ==============================
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not sleep_alarm_on and (current_time - last_sleep_alert) > COOLDOWN_TIME:
                    sleep_channel.play(sleep_sound, loops=-1)
                    sleep_alarm_on = True
                    sleep_alert_start_time = current_time
                    last_sleep_alert = current_time
                    with open("alert_log.txt", "a") as f:
                        f.write(f"[{time.strftime('%H:%M:%S')}] DROWSINESS ALERT\n")
                cv2.putText(frame, "DROWSINESS ALERT!", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                if sleep_alert_start_time:
                    duration = current_time - sleep_alert_start_time
                    cv2.putText(frame, f"Alert Time: {duration:.1f}s", (20, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            COUNTER = 0
            if sleep_alarm_on:
                sleep_channel.stop()
                sleep_alarm_on = False
                sleep_alert_start_time = None

        # ==============================
        # Yawning Detection
        # ==============================
        if mar > MAR_THRESH:
            YAWN_COUNTER += 1
            if YAWN_COUNTER >= YAWN_CONSEC_FRAMES:
                if not yawn_alarm_on and (current_time - last_yawn_alert) > COOLDOWN_TIME:
                    yawn_channel.play(yawn_sound, loops=-1)
                    yawn_alarm_on = True
                    yawn_alert_start_time = current_time
                    last_yawn_alert = current_time
                    with open("alert_log.txt", "a") as f:
                        f.write(f"[{time.strftime('%H:%M:%S')}] YAWN ALERT\n")
                cv2.putText(frame, "YAWN ALERT!", (20, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                if yawn_alert_start_time:
                    duration = current_time - yawn_alert_start_time
                    cv2.putText(frame, f"Alert Time: {duration:.1f}s", (20, 160),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            YAWN_COUNTER = 0
            if yawn_alarm_on:
                yawn_channel.stop()
                yawn_alarm_on = False
                yawn_alert_start_time = None

        # Display EAR and MAR
        cv2.putText(frame, f"EAR: {ear:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (330, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Smart Driver Monitor", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# ==============================
# Cleanup
# ==============================
cv2.destroyAllWindows()
vs.stop()
pygame.mixer.quit()
print("System stopped cleanly.")
