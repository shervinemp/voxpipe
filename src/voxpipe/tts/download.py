"""Download the weights of the configured TTS provider: ``python -m voxpipe.tts.download``."""

import sys

from ..core.config import get_config
from ..core.exceptions import VoxpipeError
from ..core.utils import get_logger, setup_logging
from .model import TTSProviders


def main() -> int:
    setup_logging(stream=sys.stdout)
    logger = get_logger(__name__)

    tts_config = get_config().tts
    try:
        TTSProviders.get(tts_config.provider).download(tts_config)
    except (VoxpipeError, ValueError, OSError) as e:
        logger.error("Failed to download model: %s", e)
        return 1
    logger.info("Model download completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
