"""Command-line entry point: ``voxpipe`` / ``python -m voxpipe``."""

import argparse
import logging
import sys
from typing import List, Optional

from .core.config import load_config, set_config
from .core.exceptions import VoxpipeError
from .core.utils import get_logger, load_specs, setup_logging
from .llm.tools import Tool


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="voxpipe", description="Run the voxpipe voice pipeline.")
    parser.add_argument(
        "specs_path",
        nargs="?",
        default=None,
        help="Optional JSON file of tool definitions (OpenAI function schema) to expose to the model.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a YAML configuration file (default: $VOXPIPE_CONFIG, else ./config.yaml if present).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level.",
    )
    parser.add_argument(
        "--push-to-talk",
        default=None,
        help="Listen only while this key combination is held, e.g. 'a', '<ctrl>+k', '<ctrl_r>+<shift_r>'. "
        "Omit for always-on listening.",
    )
    parser.add_argument(
        "--press-to-reset",
        default=None,
        help="Key combination that clears the conversation, e.g. '<ctrl_l>+<ctrl_r>'.",
    )
    return parser.parse_args(argv)


def _load_tools(specs_path: str) -> List[Tool]:
    """Read tool definitions from a JSON file holding one tool, a list of tools, or ``{"tools": [...]}``."""
    spec = load_specs(specs_path)
    if isinstance(spec, dict):
        spec = spec.get("tools", [spec])
    return [Tool.from_dict(item) for item in spec]


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    setup_logging(log_level=args.log_level)
    logger = get_logger(__name__)

    # Imported after argument parsing so that `--help` does not wait on the audio stack.
    from .pipeline.pipeline import Pipeline

    try:
        set_config(load_config(args.config))
        tools = _load_tools(args.specs_path) if args.specs_path else []
        pipeline = Pipeline.from_config(push_to_talk=args.push_to_talk, press_to_reset=args.press_to_reset)
    except ImportError as e:
        logger.critical("A required package is missing (%s). Install the matching extra, e.g. `voxpipe[all]`.", e)
        return 1
    except (VoxpipeError, OSError, ValueError) as e:
        logger.critical("Failed to start: %s", e, exc_info=logger.isEnabledFor(logging.DEBUG))
        return 1

    with pipeline:
        if tools:
            pipeline.register_tools(*tools)
            logger.info("Loaded %d tool(s) from '%s'.", len(tools), args.specs_path)
        logger.info(
            "Voice pipeline ready. %s",
            f"Hold {pipeline.push_to_talk} to speak." if pipeline.push_to_talk else "Always-on listening.",
        )
        try:
            pipeline.run()
        except KeyboardInterrupt:
            logger.info("Interrupted; shutting down.")
        except VoxpipeError as e:
            logger.critical("Pipeline stopped: %s", e)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
