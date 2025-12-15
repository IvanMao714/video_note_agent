# from typing import Optional, Union
#
# from langgraph.checkpoint.postgres import PostgresSaver
#
# from memory.providers.base import BaseMemoryClient
# from langgraph.checkpoint.base import BaseCheckpointSaver
# from log import get_logger
#
# logger = get_logger(__name__)
#
#
# class PostgresMemoryClient(BaseMemoryClient):
#     """PostgreSQL memory client implementation for LangGraph store operations."""
#
#     def __init__(
#         self,
#         connection_string: str,
#         table_name: str = "checkpoints",
#     ):
#         """Initialize PostgreSQL memory client.
#
#         Args:
#             connection_string: PostgreSQL connection string (e.g., postgresql://user:password@localhost/dbname).
#             async_mode: Whether to use async checkpoint saver. Defaults to False.
#             table_name: Name of the table to store checkpoints. Defaults to "checkpoints".
#         """
#         self.connection_string = connection_string
#         self.table_name = table_name
#
#
#
#     def get_store(self):
#         """Get the store instance (new API).
#
#         Returns:
#             PostgresStore context manager if available, None otherwise.
#
#         Note:
#             PostgresStore.from_conn_string() returns a context manager.
#             The caller should use it with a 'with' statement or enter the context.
#         """
#
#         return PostgresSaver.from_conn_string(self.connection_string)
#     def get_checkpointer(self) -> BaseCheckpointSaver:
#         """Get the checkpoint saver instance.
#
#         Returns:
#             BaseCheckpointSaver instance (PostgresSaver or AsyncPostgresSaver).
#
#         Note:
#             PostgresSaver.from_conn_string() returns a context manager.
#             This method enters the context and returns the actual checkpointer instance.
#             The context will remain open for the lifetime of the application.
#         """
from __future__ import annotations
from typing import Optional

from langgraph.store.postgres import PostgresStore
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.base import BaseStore
from langgraph.checkpoint.base import BaseCheckpointSaver

class PostgresMemoryClient:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string

        self._store_cm = None
        self._store: Optional[BaseStore] = None

        self._ckpt_cm = None
        self._checkpointer: Optional[BaseCheckpointSaver] = None

    # --- long memory store ---
    def get_store(self) -> BaseStore:
        """Return an entered PostgresStore instance (long-term memory)."""
        if self._store is None:
            self._store_cm = PostgresStore.from_conn_string(self.connection_string)
            self._store = self._store_cm.__enter__()
            self._store.setup()
        return self._store

    # --- short memory checkpoint saver ---
    def get_checkpointer(self) -> BaseCheckpointSaver:
        """Return an entered PostgresSaver instance (checkpointer)."""
        if self._checkpointer is None:
            self._ckpt_cm = PostgresSaver.from_conn_string(self.connection_string)
            self._checkpointer = self._ckpt_cm.__enter__()
            self._checkpointer.setup()
        return self._checkpointer

    def close(self) -> None:
        """Close underlying connections."""
        if self._ckpt_cm is not None:
            self._ckpt_cm.__exit__(None, None, None)
            self._ckpt_cm = None
            self._checkpointer = None
        if self._store_cm is not None:
            self._store_cm.__exit__(None, None, None)
            self._store_cm = None
            self._store = None