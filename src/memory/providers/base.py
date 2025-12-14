from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from langgraph.checkpoint.base import BaseCheckpointSaver


class BaseMemoryClient(ABC):
    """Base memory client class that defines a unified interface for checkpoint operations."""

    @abstractmethod
    def get_checkpointer(self) -> BaseCheckpointSaver:
        """Get the checkpoint saver instance.

        Returns:
            BaseCheckpointSaver instance for use with LangGraph.
        """
        pass

    @abstractmethod
    def setup(self) -> None:
        """Setup the memory storage (e.g., create tables if needed).

        This method should be idempotent and safe to call multiple times.
        """
        pass

    @abstractmethod
    def cleanup(self, thread_id: Optional[str] = None) -> None:
        """Clean up memory storage.

        Args:
            thread_id: Optional thread ID to clean up. If None, cleans up all threads.
        """
        pass
