from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union


from langgraph.store.base import BaseStore
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.base import BaseCheckpointSaver



class BaseMemoryClient(ABC):
    """Base memory client class that defines a unified interface for memory/store operations."""

    @abstractmethod
    def get_store(self) -> Union[BaseStore, BaseCheckpointSaver, None]:
        """Get the store or checkpointer instance for use with LangGraph.

        Returns:
            Store instance (BaseStore) or CheckpointSaver instance (BaseCheckpointSaver).
            Returns None if not available.
        """
        pass

    @abstractmethod
    def get_checkpointer(self) -> BaseCheckpointSaver:
        """Get the checkpoint saver instance (legacy API support).

        Returns:
            BaseCheckpointSaver instance for use with LangGraph.
        """
        pass

