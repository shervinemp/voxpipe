#!/usr/bin/env python3
"""
Setup script for LLM module.

This script handles downloading the model from the Hugging Face Hub.
"""

import sys

from . import LLMProviders

from ..core.utils import get_logger, setup_logging
from ..core.config import config


def main():
    """
    Main function to set up the LLM environment.
    """
    setup_logging(stream=sys.stdout)
    logger = get_logger(__name__)

    try:
        provider = config.get("llm.model")
        provider_cls = LLMProviders.get(provider)
        if not hasattr(provider_cls, "download"):
            raise ValueError(
                f"Provider {provider!r} does not use downloadable local weights."
            )
        provider_cls.download()
        logger.info("Model download completed successfully.")
    except Exception as e:
        logger.error(f"Failed to download model: {e}")


if __name__ == "__main__":
    main()
