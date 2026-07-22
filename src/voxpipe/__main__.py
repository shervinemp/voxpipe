import argparse
import sys

from .pipeline.pipeline import Pipeline
from .llm.tools import Tool
from .core.utils import load_specs, setup_logging, get_logger
from .core.config import config


def main():
    parser = argparse.ArgumentParser(
        description="Run the voxpipe voice pipeline."
    )
    parser.add_argument(
        "specs_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to an optional tool specification JSON file.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level.",
    )
    parser.add_argument(
        "--push-to-talk",
        type=str,
        default=None,
        help="Enable push-to-talk (ASR muted until held). "
        "Examples: 'a', '<ctrl>+k', '<shift>+<alt>+s'. "
        "Omit for always-on VAD mode.",
    )
    parser.add_argument(
        "--press-to-reset",
        type=str,
        default=None,
        help="Enable press-to-reset with the specified key or key combination. "
        "Examples: '<ctrl_l>+<ctrl_r>'. ",
    )
    args = parser.parse_args()

    setup_logging(log_level=args.log_level)
    logger = get_logger(__name__)

    tools = []
    if args.specs_path:
        try:
            tools_spec = load_specs(args.specs_path)
            if isinstance(tools_spec, list):
                tools = [Tool.from_dict(item) for item in tools_spec]
            elif isinstance(tools_spec, dict) and "tools" in tools_spec:
                tools = [Tool.from_dict(item) for item in tools_spec["tools"]]
            elif isinstance(tools_spec, dict):
                tools = [Tool.from_dict(tools_spec)]
            logger.info(f"Successfully parsed {len(tools)} tool(s) from '{args.specs_path}'.")
        except Exception as e:
            logger.critical(
                f"Error parsing API spec '{args.specs_path}': {e}. Exiting.",
                exc_info=True,
            )
            sys.exit(1)

    try:
        pipe = Pipeline(
            push_to_talk=args.push_to_talk,
            press_to_reset=args.press_to_reset,
        )
        if tools:
            pipe.register_tools(*tools)
        logger.info("Pipeline instance created.")
    except Exception as e:
        logger.critical(
            f"Failed to initialize Pipeline: {e}. Exiting.", exc_info=True
        )
        sys.exit(1)

    logger.info("Voice pipeline ready.")
    pipe.run()


if __name__ == "__main__":
    main()
