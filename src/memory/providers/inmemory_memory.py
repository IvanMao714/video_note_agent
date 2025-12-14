from langgraph.checkpoint.memory import MemorySaver

from memory.providers.base import BaseMemoryClient
from langgraph.checkpoint.base import BaseCheckpointSaver
from log import get_logger

logger = get_logger(__name__)


class InMemoryMemoryClient(BaseMemoryClient):
    """In-memory memory client implementation for LangGraph checkpoint operations.

    This is suitable for development and testing purposes only.
    Data will not persist across application restarts.
    """

    def __init__(self):
        """Initialize in-memory memory client."""
        self._checkpointer: BaseCheckpointSaver = MemorySaver()
        logger.info("In-memory checkpoint saver initialized")

    def get_checkpointer(self) -> BaseCheckpointSaver:
        """Get the checkpoint saver instance.

        Returns:
            MemorySaver instance.
        """
        return self._checkpointer

    def setup(self) -> None:
        """Setup the in-memory storage.

        No setup is needed for in-memory storage.
        """
        logger.info("In-memory checkpoint saver requires no setup")

    def cleanup(self, thread_id: str = None) -> None:
        """Clean up in-memory storage.

        Args:
            thread_id: Optional thread ID to clean up. If None, cleans up all threads.
        """
        if thread_id:
            logger.info(f"Cleanup requested for thread_id: {thread_id}")
            # MemorySaver doesn't have a direct cleanup method
            # In practice, you would need to clear the internal storage
        else:
            logger.info("Cleanup requested for all threads")
            # Clear all checkpoints
            if hasattr(self._checkpointer, "storage"):
                self._checkpointer.storage.clear()
                logger.info("All in-memory checkpoints cleared")
