"""Translate raw hand landmarks into gesture events (finger extension -> note on/off)."""

FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
_TIP = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
_PIP = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}
_THUMB_IP = 3
_THUMB_REF = 5  # index finger MCP, used as a palm-distance reference for the thumb

FINGER_TO_DEGREE = {"thumb": 0, "index": 1, "middle": 2, "ring": 3, "pinky": 4}


def _distance(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def _is_extended(landmarks, finger):
    if finger == "thumb":
        # Thumb bends sideways (not up/down) and flips direction with handedness,
        # so use distance from the palm instead of a y-comparison like the other fingers.
        tip, ip, ref = landmarks[_TIP["thumb"]], landmarks[_THUMB_IP], landmarks[_THUMB_REF]
        return _distance(tip, ref) > _distance(ip, ref)
    tip, pip = landmarks[_TIP[finger]], landmarks[_PIP[finger]]
    return tip.y < pip.y  # tip above its middle knuckle = extended


def get_finger_states(landmarks):
    """Return {finger_name: is_extended} for the current frame."""
    return {finger: _is_extended(landmarks, finger) for finger in FINGER_NAMES}


class GestureTracker:
    """Tracks per-finger extension state across frames and emits events on transitions."""

    def __init__(self):
        self._previous = {finger: False for finger in FINGER_NAMES}

    def update(self, landmarks):
        """Return a list of (finger_name, "on"|"off") for fingers that changed state this frame."""
        current = get_finger_states(landmarks)
        events = []
        for finger in FINGER_NAMES:
            if current[finger] and not self._previous[finger]:
                events.append((finger, "on"))
            elif self._previous[finger] and not current[finger]:
                events.append((finger, "off"))
        self._previous = current
        return events

    def release_all(self):
        """Return "off" events for every currently-held finger and reset state.

        Call this when the hand disappears from frame (e.g. lowered out of
        view) so notes triggered by extended fingers don't hang forever
        without a matching "off" event.
        """
        events = [(finger, "off") for finger, held in self._previous.items() if held]
        self._previous = {finger: False for finger in FINGER_NAMES}
        return events


def finger_to_scale_degree(finger):
    """Map a finger name to its pentatonic scale degree index (0-4)."""
    return FINGER_TO_DEGREE[finger]
