#!/usr/bin/env python3
"""
Setup script for Text-to-Speech (TTS) system using Kokoro-ONNX.

This script handles installing dependencies and setting up the environment.
"""

import sys

from .model import TTSProviders

from ..core.config import config
from ..core.utils import get_logger, setup_logging


def main():
    """
    Main function to set up the TTS environment.
    """
    setup_logging(stream=sys.stdout)
    logger = get_logger(__name__)

    try:
        provider = config.get("tts.provider")
        getattr(TTSProviders, provider).download()
        logger.info("Model download completed successfully.")
    except Exception as e:
        logger.error(f"Failed to download model: {e}")


if __name__ == "__main__":
    main()
