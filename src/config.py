"""Tunable constants: root note, scale type, camera index, sample rate, gesture thresholds."""
import os

MAJOR_SCALE = [0, 2, 4, 7, 9]  # Major scale intervals
MINOR_SCALE = [0, 3, 5, 7, 10]  # Minor scale intervals

CHROMATIC_SCALE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

ROOT_NOTE = "C"
SCALE_TYPE = "major"
OCTAVE = 4
CAMERA_INDEX = 1  # 0 is the Continuity Camera (iPhone) on this machine and drops frames; 1 is the built-in webcam
# AVFoundation on macOS can take a few frames to warm up after opening, so a
# read failure isn't necessarily fatal until it persists this many frames in a row.
MAX_CONSECUTIVE_FRAME_FAILURES = 30

# Resolved relative to this file so it works regardless of the CWD main.py is run from.
HAND_LANDMARKER_MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

SAMPLE_RATE = 44100
BLOCK_SIZE = 256
MAX_POLYPHONY = 5  # one voice per finger

# ADSR envelope, in seconds (sustain level is 0-1, not a duration)
ATTACK_TIME = 0.02
DECAY_TIME = 0.1
SUSTAIN_LEVEL = 0.6
RELEASE_TIME = 0.3
