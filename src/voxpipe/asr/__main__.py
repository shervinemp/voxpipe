"""Print what the configured recogniser hears: ``python -m voxpipe.asr``."""

import argparse
import sys

from ..core.config import get_config
from ..core.utils import get_logger, setup_logging
from .model import ASRProviders


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time, continuous ASR from the microphone.")
    parser.add_argument(
        "--sound-device",
        default=None,
        help="Input device index or name fragment (default: asr.sound_device from the configuration). "
        "Use `python -m sounddevice` to list devices.",
    )
    args = parser.parse_args()

    setup_logging(log_level="DEBUG", stream=sys.stdout)
    logger = get_logger(__name__)

    config = get_config()
    if args.sound_device is not None:
        device = int(args.sound_device) if args.sound_device.isdigit() else args.sound_device
        config = config.model_copy(update={"asr": config.asr.model_copy(update={"sound_device": device})})

    logger.info("Starting ASR...")
    with ASRProviders.create(config=config) as recogniser:
        try:
            for text in recogniser:
                logger.info(text)
        except KeyboardInterrupt:
            logger.info("Stopping ASR...")


if __name__ == "__main__":
    main()
