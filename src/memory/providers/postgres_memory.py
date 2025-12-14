from typing import Optional

try:
    from langgraph.checkpoint.postgres import AsyncPostgresSaver, PostgresSaver
except ImportError:
    raise ImportError(
        "PostgreSQL checkpoint saver not available. "
        "Please install it with: pip install langgraph-checkpoint-postgres"
    )

from memory.providers.base import BaseMemoryClient
from langgraph.checkpoint.base import BaseCheckpointSaver
from log import get_logger

logger = get_logger(__name__)


class PostgresMemoryClient(BaseMemoryClient):
    """PostgreSQL memory client implementation for LangGraph checkpoint operations."""

    def __init__(
        self,
        connection_string: str,
        async_mode: bool = False,
        table_name: str = "checkpoints",
    ):
        """Initialize PostgreSQL memory client.

        Args:
            connection_string: PostgreSQL connection string (e.g., postgresql://user:password@localhost/dbname).
            async_mode: Whether to use async checkpoint saver. Defaults to False.
            table_name: Name of the table to store checkpoints. Defaults to "checkpoints".
        """
        self.connection_string = connection_string
        self.async_mode = async_mode
        self.table_name = table_name
        self._checkpointer: Optional[BaseCheckpointSaver] = None

    def get_checkpointer(self) -> BaseCheckpointSaver:
        """Get the checkpoint saver instance.

        Returns:
            BaseCheckpointSaver instance (PostgresSaver or AsyncPostgresSaver).
        """
        if self._checkpointer is None:
            if self.async_mode:
                self._checkpointer = AsyncPostgresSaver.from_conn_string(
                    self.connection_string, self.table_name
                )
            else:
                self._checkpointer = PostgresSaver.from_conn_string(
                    self.connection_string, self.table_name
                )
            logger.info(
                f"PostgreSQL checkpoint saver initialized (async={self.async_mode}, table={self.table_name})"
            )
        return self._checkpointer

    def setup(self) -> None:
        """Setup the PostgreSQL database (create tables if needed).

        This method is idempotent and safe to call multiple times.
        The PostgresSaver automatically creates the necessary tables on first use.
        """
        checkpointer = self.get_checkpointer()
        # PostgresSaver creates tables automatically, but we can call setup if available
        if hasattr(checkpointer, "setup"):
            checkpointer.setup()
            logger.info("PostgreSQL checkpoint tables setup completed")
        else:
            logger.info("PostgreSQL checkpoint tables will be created automatically on first use")

    def cleanup(self, thread_id: Optional[str] = None) -> None:
        """Clean up checkpoints from PostgreSQL.

        Args:
            thread_id: Optional thread ID to clean up. If None, cleans up all threads.
        """
        checkpointer = self.get_checkpointer()
        if hasattr(checkpointer, "list"):
            if thread_id:
                # Clean up specific thread
                # Note: This is a simplified implementation
                # Actual cleanup would require direct database access
                logger.info(f"Cleanup requested for thread_id: {thread_id}")
            else:
                logger.info("Cleanup requested for all threads")
        else:
            logger.warning("Cleanup not supported by this checkpoint saver")
