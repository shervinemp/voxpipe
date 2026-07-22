#!/usr/bin/env python3
"""
Main entry point for the Text-to-Speech (TTS) system.

This script demonstrates how to set up and run a continuous TTS loop that gets text from user input.
"""

import sys

from ..core.utils import setup_logging, get_logger

from .model import TTS


def main():
    """
    Main function to run a continuous TTS loop.
    """
    setup_logging(log_level="DEBUG")
    logger = get_logger(__name__)

    try:
        tts = TTS()
        tts.start()

        logger.info("Starting continuous TTS loop. Type 'exit' to quit.")
        while True:
            try:
                user_input = input("Enter text to speak (or 'exit' to quit): ")

                if user_input.lower() == "exit":
                    logger.info("Exiting TTS loop...")
                    break

                _ = tts(user_input)
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt detected. Exiting...")
                break
            except Exception as e:
                logger.error(f"Error in TTS processing: {e}")

    except Exception as e:
        logger.error(f"Error in main(): {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
