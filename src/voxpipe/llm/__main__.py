"""Smoke-test the configured LLM with two demo tools: ``python -m voxpipe.llm``."""

import sys

from ..core.config import get_config
from ..core.utils import get_logger, setup_logging
from . import Conversation, LLMProviders, Session, Tool


def get_current_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather in a given location.

    Args:
        location: The city and state, e.g. San Francisco, CA.
        unit: The temperature unit to use, celsius or fahrenheit. Infer this from the user's location.
    """
    return "Rainy"


def get_stock_price(stock: str) -> int:
    """Get the current stock price.

    Args:
        stock: The stock symbol.
    """
    return 100


def main() -> None:
    setup_logging(log_level="INFO", stream=sys.stdout)
    logger = get_logger(__name__)

    llm_config = get_config().llm
    llm = LLMProviders.create(llm_config.backend, llm_config.model)

    conversation = Conversation()
    for name, function in (("get_current_weather", get_current_weather), ("get_stock_price", get_stock_price)):
        conversation.tools[name] = Tool.from_callable(name, function)
    conversation.set_system_message(
        "You are a helpful assistant. "
        "You can answer questions, provide information, and assist with various tasks. "
        "If you don't know the answer, you can say 'I don't know'."
    )

    prompt = "What is the weather like in Beijing now and what's the stock price of NVDA?"
    logger.info("Prompt: %s", prompt)
    with Session(llm=llm, conversation=conversation) as session:
        logger.info("Response: %s", "".join(session(prompt)))


if __name__ == "__main__":
    main()
