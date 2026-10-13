"""Speak lines typed at the prompt with the configured TTS provider: ``python -m voxpipe.tts``."""

import time

from ..core.utils import get_logger, setup_logging
from .model import TTSProviders


def main() -> None:
    setup_logging(log_level="DEBUG")
    logger = get_logger(__name__)

    with TTSProviders.create() as tts:
        logger.info("Type text to speak, or 'exit' to quit.")
        try:
            while (line := input("> ")).strip().lower() != "exit":
                tts(line)
        except EOFError:
            pass
        except KeyboardInterrupt:
            return
        try:
            while tts.is_speaking:  # leaving the block stops playback at once
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
