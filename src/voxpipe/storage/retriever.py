"""Abstract interface for context retrieval and storage."""

from abc import ABC, abstractmethod


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3,
                 **kwargs) -> list[dict]:
        """Result dicts with at least ``{"content": str}``."""

    def store(self, content: str, **kwargs):
        pass

    def close(self):
        pass

    def get_state(self) -> str:
        return ""
