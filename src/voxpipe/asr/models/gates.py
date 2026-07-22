"""Audio gates — pre-filters that decide whether to forward audio to the VAD.

A gate is any callable/object with a ``process(chunk: np.ndarray) -> str``
method returning ``"active"`` (forward audio) or another string (keep
dropping). Examples:

* Wake-word detector (Porcupine): returns ``"active"`` when the keyword
  is spoken, ``"idle"`` otherwise.
* Noise gate: returns ``"active"`` only when RMS exceeds a threshold.
* Push-to-talk: returns ``"active"`` when a key is held.
"""
