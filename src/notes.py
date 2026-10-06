"""Pentatonic scale definition and scale-degree -> frequency mapping."""
from config import MAJOR_SCALE, MINOR_SCALE, CHROMATIC_SCALE
def get_pentatonic_scale(root_note: str, scale_type: str) -> list:
    root_index = CHROMATIC_SCALE.index(root_note)
    if scale_type == "major":
        scale_intervals = MAJOR_SCALE
    else:
        scale_intervals = MINOR_SCALE

    scale_notes =[]
    for interval in scale_intervals:
        note_index = (root_index + interval) % len(CHROMATIC_SCALE)
        scale_notes.append(CHROMATIC_SCALE[note_index])
    return scale_notes

def get_frequency(note: str, octave: int) -> float:
    A4_FREQ = 440.0
    A4_INDEX = CHROMATIC_SCALE.index("A") + 12 * 4
    note_index = CHROMATIC_SCALE.index(note) + 12 * octave
    semitone_diff = note_index - A4_INDEX
    frequency = A4_FREQ * (2 ** (semitone_diff / 12))
    return frequency
