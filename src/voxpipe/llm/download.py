"""Download the weights of the configured local LLM: ``python -m voxpipe.llm.download``."""

import sys

from ..core.config import get_config
from ..core.downloader import ensure_downloaded
from ..core.exceptions import VoxpipeError
from ..core.utils import get_logger, setup_logging


def main() -> int:
    setup_logging(stream=sys.stdout)
    logger = get_logger(__name__)

    llm_config = get_config().llm
    entry = llm_config.local.get(llm_config.model)
    if entry is None:
        logger.error("No local model is configured under llm.local.%s.", llm_config.model)
        return 1
    if entry.model_path:
        logger.info("%s uses the local file %s; nothing to download.", llm_config.model, entry.model_path)
        return 0

    try:
        ensure_downloaded(llm_config.model)
    except (VoxpipeError, ValueError, OSError) as e:
        logger.error("Failed to download model: %s", e)
        return 1
    logger.info("Model download completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
