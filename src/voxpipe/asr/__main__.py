#!/usr/bin/env python3
"""
Main entry point for the Automatic Speech Recognition (ASR) system.

This script demonstrates how to set up and run the ASR pipeline.
"""

import sys

from .model import ASRProviders

from ..core.utils import setup_logging, get_logger


def parse_asr_args():
    """Parses command-line arguments for the ASR system."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Real-time, continuous ASR from microphone using ONNX models and VAD."
    )

    parser.add_argument(
        "--sound-device",
        type=int,
        default=0,
        help="Specific audio input device ID. "
        "Use `python -m sounddevice` to list devices.",
    )
    return parser.parse_args()


def main():
    """
    Main function to run the ASR system.
    """
    setup_logging(log_level="DEBUG", stream=sys.stdout)
    logger = get_logger(__name__)

    args = parse_asr_args().__dict__
    model = ASRProviders.ParakeetV2(**args)

    logger.info("Starting ASR...")
    model.start()

    try:
        for text in model:
            logger.info(text)
    except KeyboardInterrupt:
        logger.info("Stopping ASR...")
        model.stop()


if __name__ == "__main__":
    main()
