import pytest

from oss.oss import get_oss_by_type


@pytest.fixture
def minios_client():
    return get_oss_by_type("minio")


def test_get_oss_by_type_minio(minios_client):
    assert minios_client.bucket == "test"

def test_upload_file(minios_client):
    # Create a temporary file to upload
    success = minios_client.upload_file(
        bucket_name="test",
        object_name="test.txt",
        file_path="test.txt",
    )
    assert success is True

def test_upload_bytes(minios_client):
    data = b"this is unit test file for oss"
    success = minios_client.upload_bytes(
        bucket_name="test",
        object_name="test_bytes.txt",
        data=data,
    )
    assert success is True

def test_download_file(minios_client):
    data = minios_client.download_file(
        bucket_name="notes",
        object_name="cs336/video/cs336_01.json",
    )
    print(data)
    assert data == b"this is unit test file for oss"

