#!/usr/bin/env python3
"""
Main entry point for the LLM module.

This script tests the language model functionality.
"""

import sys
from . import Conversation, Session, Tool, LLMProviders
from ..core.utils import setup_logging, get_logger
from ..core.config import config


def main():
    """
    Main function to test the LLM module.
    """
    setup_logging(log_level="INFO", stream=sys.stdout)
    logger = get_logger(__name__)

    tools = list(
        map(
            Tool.from_dict,
            [
                {
                    "type": "function",
                    "function": {
                        "name": "get_current_weather",
                        "description": "Get the current weather in a given "
                        "location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city and state, e.g. San Francisco, CA",
                                },
                                "unit": {
                                    "type": "string",
                                    "description": "The temperature unit to use. "
                                    "Infer this from the users location.",
                                    "enum": ["celsius", "fahrenheit"],
                                },
                            },
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_stock_price",
                        "description": "Get the current stock price",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "stock": {
                                    "type": "string",
                                    "description": "The stock symbol",
                                },
                            },
                        },
                    },
                },
            ],
        )
    )
    tools[0].callback = lambda **kwargs: "Rainy"
    tools[1].callback = lambda **kwargs: 100
    backend = config.get("llm.backend")
    model = config.get("llm.model")
    llm = LLMProviders.create(backend, model)
    conversation = Conversation()
    conversation.tools = tools
    conversation.set_system_message(
        (
            "You are a helpful assistant."
            "You can answer questions, provide information, and assist with various tasks."
            "If you don't know the answer, you can say 'I don't know'."
        )
    )

    session = Session(llm=llm, conversation=conversation)
    logger.info("Session initialized successfully.")

    prompt = "What is the weather like in Beijing now and what's the stock price of NVDA?"
    logger.info(f"Prompt: {prompt}")

    try:
        response = "".join(session(prompt))
        logger.info(f"Response: {response}")
    finally:
        session.close()


if __name__ == "__main__":
    main()
