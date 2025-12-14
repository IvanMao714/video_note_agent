import pytest

from memory.memory import get_memory_by_type, get_memory_client, get_configured_memory_clients
from memory.providers.base import BaseMemoryClient
from langgraph.checkpoint.base import BaseCheckpointSaver


@pytest.fixture
def inmemory_client():
    """Fixture for in-memory memory client."""
    return get_memory_by_type("inmemory")


@pytest.fixture
def postgres_client():
    """Fixture for PostgreSQL memory client (requires configuration)."""
    try:
        return get_memory_by_type("postgres")
    except (ValueError, ImportError) as e:
        pytest.skip(f"PostgreSQL memory client not available: {e}")


def test_get_memory_by_type_inmemory(inmemory_client):
    """Test getting in-memory memory client."""
    assert inmemory_client is not None
    assert isinstance(inmemory_client, BaseMemoryClient)


def test_get_memory_by_type_postgres(postgres_client):
    """Test getting PostgreSQL memory client."""
    assert postgres_client is not None
    assert isinstance(postgres_client, BaseMemoryClient)


def test_get_memory_client_default():
    """Test getting default memory client."""
    client = get_memory_client()
    assert client is not None
    assert isinstance(client, BaseMemoryClient)


def test_get_checkpointer_inmemory(inmemory_client):
    """Test getting checkpoint saver from in-memory client."""
    checkpointer = inmemory_client.get_checkpointer()
    assert checkpointer is not None
    assert isinstance(checkpointer, BaseCheckpointSaver)


def test_get_checkpointer_postgres(postgres_client):
    """Test getting checkpoint saver from PostgreSQL client."""
    checkpointer = postgres_client.get_checkpointer()
    assert checkpointer is not None
    assert isinstance(checkpointer, BaseCheckpointSaver)


def test_setup_inmemory(inmemory_client):
    """Test setup method for in-memory client."""
    inmemory_client.setup()


def test_setup_postgres(postgres_client):
    """Test setup method for PostgreSQL client."""
    postgres_client.setup()


def test_cleanup_inmemory(inmemory_client):
    """Test cleanup method for in-memory client."""
    inmemory_client.cleanup()


def test_cleanup_postgres(postgres_client):
    """Test cleanup method for PostgreSQL client."""
    postgres_client.cleanup()


def test_cleanup_with_thread_id_inmemory(inmemory_client):
    """Test cleanup with specific thread ID for in-memory client."""
    thread_id = "test-thread-id"
    inmemory_client.cleanup(thread_id=thread_id)


def test_cleanup_with_thread_id_postgres(postgres_client):
    """Test cleanup with specific thread ID for PostgreSQL client."""
    thread_id = "test-thread-id"
    postgres_client.cleanup(thread_id=thread_id)


def test_get_configured_memory_clients():
    """Test getting all configured memory clients."""
    clients = get_configured_memory_clients()
    assert isinstance(clients, dict)


def test_memory_client_caching():
    """Test that memory clients are cached."""
    client1 = get_memory_by_type("inmemory")
    client2 = get_memory_by_type("inmemory")
    assert client1 is client2


def test_invalid_memory_type():
    """Test that invalid memory type raises ValueError."""
    with pytest.raises(ValueError):
        get_memory_by_type("invalid_type")  # type: ignore
