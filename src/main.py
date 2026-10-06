"""Entry point: wires capture -> gesture -> note -> synth into the main loop."""
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

import config
import notes
from gestures import GestureTracker, finger_to_scale_degree
from synth import Synth

hand_connections = vision.HandLandmarksConnections.HAND_CONNECTIONS


def play_event(synth, scale, finger, event):
    degree = finger_to_scale_degree(finger)
    note = scale[degree]
    frequency = notes.get_frequency(note, config.OCTAVE)
    if event == "on":
        synth.note_on(finger, frequency)
    else:
        synth.note_off(finger)


def draw_hand(frame, hand_landmarks):
    vision.drawing_utils.draw_landmarks(frame, hand_landmarks, hand_connections)


def main():
    scale = notes.get_pentatonic_scale(config.ROOT_NOTE, config.SCALE_TYPE)

    synth = Synth()
    tracker = GestureTracker()

    cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=config.HAND_LANDMARKER_MODEL_PATH),
        num_hands=1,
        running_mode=vision.RunningMode.VIDEO,
    )

    synth.start()
    start_time = time.monotonic()
    try:
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            consecutive_failures = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures >= config.MAX_CONSECUTIVE_FRAME_FAILURES:
                        print("Error: could not read frame.")
                        break
                    continue
                consecutive_failures = 0

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int((time.monotonic() - start_time) * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if result.hand_landmarks:
                    hand_landmarks = result.hand_landmarks[0]
                    draw_hand(frame, hand_landmarks)
                    events = tracker.update(hand_landmarks)
                else:
                    events = tracker.release_all()

                for finger, event in events:
                    play_event(synth, scale, finger, event)

                cv2.imshow("Gesture Synth", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        for finger, event in tracker.release_all():
            play_event(synth, scale, finger, event)
        synth.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
