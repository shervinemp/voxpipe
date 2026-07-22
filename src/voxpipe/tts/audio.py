import atexit
from queue import Empty, Full, Queue
import threading
import numpy as np
import sounddevice as sd

from ..core.utils import get_logger


class AudioPlayer:

    def __init__(self, output_device: int | None = None):
        self.logger = get_logger(__name__)

        if output_device is None:
            output_device = sd.default.device[1]

        self.output_device = output_device
        device_name = sd.query_devices(output_device)["name"]
        self.logger.info(
            f"AudioPlayer initialized. Using device: '{device_name}' (ID: {output_device})"
        )

        self._queue = Queue(maxsize=32)
        self._queue_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._running = False
        self._gen = 0
        atexit.register(self.stop)

    def _run(self):
        while self._running:
            try:
                audio_data, sample_rate, gen, _ = self._queue.get(timeout=1.0)
            except Empty:
                continue

            if gen != self._gen:
                continue

            self.logger.debug("AudioPlayer playing %d samples at %dHz (gen=%d)", len(audio_data), sample_rate, gen)
            sd.play(audio_data, samplerate=sample_rate, device=self.output_device, blocking=True)

    def __call__(
        self,
        audio_data: np.ndarray[np.float32 | np.int16],
        sample_rate: int,
        interrupt: bool = False,
    ):
        if interrupt:
            with self._queue_lock:
                while not self._queue.empty():
                    try:
                        self._queue.get_nowait()
                    except Empty:
                        break
        gen = self._gen
        try:
            self._queue.put_nowait((audio_data, sample_rate, gen, interrupt))
        except Full:
            self.logger.warning("TTS queue full, dropping sentence to stay responsive.")

    def stop_playback(self):
        with self._queue_lock:
            self._gen += 1

    def play(
        self,
        audio_data: np.ndarray[np.float32 | np.int16],
        sample_rate: int,
    ):
        if audio_data.dtype != np.float32:
            audio_data = (
                audio_data.astype(np.float32) / 32767.0
                if audio_data.dtype == np.int16
                else audio_data.astype(np.float32)
            )
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data /= (np.max(np.abs(audio_data)) + 1e-8)

        sd.play(
            audio_data,
            samplerate=sample_rate,
            device=self.output_device,
            blocking=True,
        )

    def start(self):
        if not self._running:
            self._running = True
            self._thread.start()

    def stop(self):
        if self._running:
            with self._queue_lock:
                while not self._queue.empty():
                    try:
                        self._queue.get_nowait()
                    except Empty:
                        break
            self._running = False
            self._thread.join(timeout=5)
            sd.stop()


def main():
    from time import sleep
    logger = get_logger("AudioPlayerExample")

    player = AudioPlayer()

    logger.info(
        "--- Testing interrupt playback with a generated sine wave ---"
    )
    sample_rate = 44100
    frequency = 440  # A4 note
    duration = 2.0  # seconds
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    sine_wave = 0.5 * np.sin(2 * np.pi * frequency * t)

    player(sine_wave, sample_rate)
    sleep(duration)

    logger.info("Sine wave playback complete.\n")


if __name__ == "__main__":
    main()
