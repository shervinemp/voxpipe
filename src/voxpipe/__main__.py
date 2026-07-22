import argparse
import sys

from .pipeline.pipeline import Pipeline

from .bridge.tool_client import ToolClient

from .core.utils import load_specs, setup_logging, get_logger
from .core.config import config


def main():
    parser = argparse.ArgumentParser(
        description="Run the voice-control pipeline."
    )
    parser.add_argument(
        "specs_path",
        type=str,
        help="Path to the tool specification JSON file.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address used by the RPC server. Non-loopback hosts require RPC_AUTH_TOKEN.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port used by the RPC server.",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default="tcp",
        choices=["tcp", "ipc"],
        help="Protocol used by the RPC server.",
    )
    parser.add_argument(
        "--tools-host",
        type=str,
        default="127.0.0.1",
        help="Host address of the tools server.",
    )
    parser.add_argument(
        "--tools-port",
        type=int,
        default=8080,
        help="Port used by the tools server.",
    )
    parser.add_argument(
        "--tools-protocol",
        type=str,
        default="tcp",
        choices=["tcp", "ipc"],
        help="Protocol used by the tools server.",
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
    parser.add_argument(
        "--gui",
        action="store_true",
        default=False,
        help="Show the transparent mic overlay button.",
    )
    args = parser.parse_args()

    setup_logging(log_level=args.log_level)
    logger = get_logger(__name__)

    try:
        tools_spec = load_specs(args.specs_path)
        logger.info(f"Successfully parsed tool spec from '{args.specs_path}'.")
    except Exception as e:
        logger.critical(
            f"Error parsing API spec '{args.specs_path}': {e}. Exiting.",
            exc_info=True,
        )
        sys.exit(1)

    try:
        tools_auth_token = config.get("tools_server.auth_token")
        tool_client = ToolClient(
            f"{args.tools_protocol}://{args.tools_host}:{args.tools_port}",
            auth_token=tools_auth_token,
        )
        tools = tool_client.from_spec(tools_spec)
    except Exception as e:
        logger.critical(
            f"Error creating tools: {e}. Exiting.",
            exc_info=True,
        )
        sys.exit(1)

    try:
        pipe = Pipeline(
            server_endpoint=f"{args.protocol}://{args.host}:{args.port}",
            push_to_talk=args.push_to_talk,
            press_to_reset=args.press_to_reset,
        )
        pipe.session.conversation.tools = tools
        logger.info("Pipeline instance created.")
    except Exception as e:
        logger.critical(
            f"Failed to initialize Pipeline: {e}. Exiting.", exc_info=True
        )
        sys.exit(1)

    try:
        from .rag.backends import create_backend
        from .rag.embeddings import Embedder
        from .rag.model import SPathRAG

        backend = create_backend()
        backend.verify_connectivity()
        embedder = Embedder()
        rag = SPathRAG(llm=pipe.session.llm, backend=backend, embedder=embedder, web_search=True)
        pipe.rag = rag
        logger.info("RAG initialized with S-Path-RAG.")
    except Exception as e:
        logger.warning(
            f"Skipping storage backend initialization: {e}"
        )

    logger.info("Voice pipeline ready.")

    if args.gui:
        try:
            from .gui import MicButton
            import threading
            gui = MicButton(pipe)
            threading.Thread(target=gui.run, daemon=True).start()
        except Exception as e:
            logger.warning("Failed to start GUI overlay: %s", e)

    pipe.run()


if __name__ == "__main__":
    main()
