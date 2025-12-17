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

    # --- short memory checkpoint saver (synchronous) ---
    def get_checkpointer(self) -> BaseCheckpointSaver:
        """Return an entered PostgresSaver instance (synchronous checkpointer)."""
        if self._checkpointer is None:
            self._ckpt_cm = PostgresSaver.from_conn_string(self.connection_string)
            self._checkpointer = self._ckpt_cm.__enter__()
            self._checkpointer.setup()
        return self._checkpointer

    def close(self) -> None:
        """Close underlying connections (synchronous)."""
        if self._ckpt_cm is not None:
            self._ckpt_cm.__exit__(None, None, None)
            self._ckpt_cm = None
            self._checkpointer = None
        if self._store_cm is not None:
            self._store_cm.__exit__(None, None, None)
            self._store_cm = None
            self._store = None
