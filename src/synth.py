"""Oscillator + ADSR envelope + real-time polyphonic audio output stream."""
import sys
import threading

import numpy as np
import sounddevice as sd

from config import (
    SAMPLE_RATE,
    BLOCK_SIZE,
    MAX_POLYPHONY,
    ATTACK_TIME,
    DECAY_TIME,
    SUSTAIN_LEVEL,
    RELEASE_TIME,
)


class _Voice:
    """One sounding note: a phase-accumulated sine oscillator plus its ADSR envelope."""

    def __init__(self, frequency, sample_rate):
        self.frequency = frequency
        self.sample_rate = sample_rate
        self.phase = 0.0
        self.samples_since_on = 0
        self.samples_since_off = 0
        self.released = False

    def note_off(self):
        self.released = True
        self.samples_since_off = 0

    def is_finished(self):
        release_samples = max(1, int(RELEASE_TIME * self.sample_rate))
        return self.released and self.samples_since_off >= release_samples

    def _envelope(self, n):
        attack_samples = max(1, int(ATTACK_TIME * self.sample_rate))
        decay_samples = max(1, int(DECAY_TIME * self.sample_rate))
        release_samples = max(1, int(RELEASE_TIME * self.sample_rate))

        if self.released:
            t = self.samples_since_off + np.arange(n)
            return np.clip(SUSTAIN_LEVEL * (1 - t / release_samples), 0.0, None)

        t = self.samples_since_on + np.arange(n)
        return np.where(
            t < attack_samples,
            t / attack_samples,
            np.where(
                t < attack_samples + decay_samples,
                1.0 + (t - attack_samples) / decay_samples * (SUSTAIN_LEVEL - 1.0),
                SUSTAIN_LEVEL,
            ),
        )

    def render(self, n):
        """Return n samples of this voice's waveform with its envelope applied."""
        t = np.arange(n) / self.sample_rate
        wave = np.sin(self.phase + 2 * np.pi * self.frequency * t) * self._envelope(n)

        self.phase = (self.phase + 2 * np.pi * self.frequency * n / self.sample_rate) % (2 * np.pi)
        self.samples_since_on += n
        if self.released:
            self.samples_since_off += n
        return wave


class Synth:
    """Polyphonic real-time synth: one Voice per active note, mixed and streamed to the audio output."""

    def __init__(self, sample_rate=SAMPLE_RATE, blocksize=BLOCK_SIZE):
        self.sample_rate = sample_rate
        self._voices = {}  # key (e.g. finger name) -> _Voice
        self._lock = threading.Lock()
        self._stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            blocksize=blocksize,
            callback=self._callback,
        )

    def start(self):
        self._stream.start()

    def stop(self):
        self._stream.stop()
        self._stream.close()

    def note_on(self, key, frequency):
        """Start (or retrigger) the voice identified by `key` at `frequency` Hz."""
        with self._lock:
            self._voices[key] = _Voice(frequency, self.sample_rate)

    def note_off(self, key):
        """Begin the release phase for the voice identified by `key`, if it's active."""
        with self._lock:
            voice = self._voices.get(key)
            if voice is not None:
                voice.note_off()

    def _callback(self, outdata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)

        mix = np.zeros(frames)
        with self._lock:
            for key, voice in list(self._voices.items()):
                mix += voice.render(frames)
                if voice.is_finished():
                    del self._voices[key]

        # Fixed normalization (not divided by current voice count) so volume
        # stays consistent as a chord is built up or released, instead of pumping.
        outdata[:, 0] = np.clip(mix / MAX_POLYPHONY, -1.0, 1.0)
