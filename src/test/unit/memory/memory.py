import pytest

from memory.memory import get_memory_by_type, get_memory_client, get_configured_memory_clients
from memory.providers.base import BaseMemoryClient
from langgraph.checkpoint.base import BaseCheckpointSaver


# @pytest.fixture
# def inmemory_client():
#     """Fixture for in-memory memory client."""
#     return get_memory_by_type("inmemory")


@pytest.fixture
def postgres_client():
    """Fixture for PostgreSQL memory client (requires configuration)."""
    return get_memory_by_type("postgres")


def test_get_memory_by_type(postgres_client):
    """Test retrieval of memory clients by type."""
    # assert isinstance(inmemory_client, BaseMemoryClient)
    assert isinstance(postgres_client, BaseMemoryClient)


def test_get_store(postgres_client):
    """Test getting the store from PostgreSQL memory client."""
    store = postgres_client.get_store()
    assert store is not None


