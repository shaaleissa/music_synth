"""Webcam capture + MediaPipe hand landmark detection (Tasks API)."""
import time

import cv2
import mediapipe as mp
import pandas as pd
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

import config

hand_connections = vision.HandLandmarksConnections.HAND_CONNECTIONS

options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=config.HAND_LANDMARKER_MODEL_PATH),
    num_hands=2,
    running_mode=vision.RunningMode.VIDEO,
)

#start webcam capture
cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

with vision.HandLandmarker.create_from_options(options) as landmarker:
    start_time = time.monotonic()
    consecutive_failures = 0
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            consecutive_failures += 1
            if consecutive_failures >= config.MAX_CONSECUTIVE_FRAME_FAILURES:
                print("Error: Could not read frame.")
                break
            continue
        consecutive_failures = 0

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.monotonic() - start_time) * 1000)

        # Process the frame with the HandLandmarker task
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Draw the hand annotations on the image.
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                vision.drawing_utils.draw_landmarks(frame, hand_landmarks, hand_connections)

                # Extract landmark coordinates and save to CSV
                landmark_data = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
                df = pd.DataFrame(landmark_data, columns=['x', 'y', 'z'])
                df.to_csv('hand_landmarks.csv', index=False)

        # Display the resulting frame
        cv2.imshow('Webcam Feed', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
