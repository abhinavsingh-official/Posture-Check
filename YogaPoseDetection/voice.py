"""Offline voice feedback via pyttsx3 - no API/internet dependency, so
it works the same as everything else here: clone and run, nothing to
configure. On a cooldown so it announces a hint once, not every frame."""

import time

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

COOLDOWN_SECONDS = 3.0


class VoiceFeedback:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and pyttsx3 is not None
        self._engine = pyttsx3.init() if self.enabled else None
        self._last_said: str | None = None
        self._last_said_at: float = 0.0

    def say(self, message: str) -> None:
        if not self.enabled or not message:
            return
        now = time.monotonic()
        if message == self._last_said and (now - self._last_said_at) < COOLDOWN_SECONDS:
            return
        self._last_said = message
        self._last_said_at = now
        self._engine.say(message)
        self._engine.runAndWait()
