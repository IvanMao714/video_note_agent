import json
from typing import Optional

from minio import Minio, S3Error

from src.oss.providers.base import BaseOSSClient
from src.log import get_logger
logger = get_logger(__name__)

class MinIOClient(BaseOSSClient):
    """MinIO client implementation for object storage operations."""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        secure: bool = False,
        region: Optional[str] = None,
        bucket_name: Optional[str] = None,
    ):
        """Initialize MinIO client.

        Args:
            endpoint: MinIO server address (e.g., localhost:9000).
            access_key: Access key ID for authentication.
            secret_key: Secret access key for authentication.
            secure: Whether to use HTTPS. Defaults to False.
            region: Region name. Optional.
            bucket_name: Default bucket name. Optional. If provided and the bucket
                doesn't exist, it will be created automatically.
        """
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.secure = secure
        self.region = region
        self.bucket = bucket_name

        self.client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
            region=region,
        )

        # Public-read policy (for DashScope to download uploaded media).
        # NOTE: Upload still requires credentials; we only grant anonymous GetObject.
        self.policy = self._build_public_read_policy(self.bucket) if self.bucket else None

        # Delay bucket check to avoid blocking network calls during initialization.
        # Bucket will be created automatically on first use if needed.
        self._bucket_checked = False

        logger.debug(f"MinIO client initialized for endpoint: {endpoint}")

    def _ensure_bucket_exists(self, bucket_name: Optional[str] = None) -> None:
        """Ensure bucket exists, create it if it doesn't exist.
        
        This method is deferred to avoid blocking network calls during initialization.
        It is called only on first use, subsequent calls will skip the check.
        
        Args:
            bucket_name: Name of the bucket to check. If None, uses self.bucket.
        """
        if self._bucket_checked:
            return
        
        bucket = bucket_name or self.bucket
        if not bucket:
            return
        
        try:
            if not self.client.bucket_exists(bucket):
                self.client.make_bucket(bucket)
                logger.warning(f"Bucket '{bucket}' did not exist and was created.")

            # Always (re)apply public-read policy on first use, even if bucket already existed.
            # This avoids 403 for anonymous downloads (e.g., DashScope file_url fetch).
            policy = self._build_public_read_policy(bucket)
            self.client.set_bucket_policy(bucket, json.dumps(policy))
            self._bucket_checked = True
        except Exception as e:
            logger.warning(f"Failed to check/create bucket '{bucket}': {e}. Will retry on next operation.")

    @staticmethod
    def _build_public_read_policy(bucket: str) -> dict:
        """Build a public-read bucket policy for the given bucket name."""
        return {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"AWS": "*"},
                    "Action": ["s3:GetObject"],
                    "Resource": f"arn:aws:s3:::{bucket}/*",
                }
            ],
        }

#
    def upload_file(self, bucket_name: str, object_name: str, file_path: str) -> bool:
        """Upload a file to MinIO.
        
        Args:
            bucket_name: Name of the bucket where the file will be stored.
            object_name: Object name (path of the file in MinIO).
            file_path: Local file path to upload.
            
        Returns:
            True if upload is successful, False otherwise.
        """
        # Ensure bucket exists (deferred check)
        self._ensure_bucket_exists(bucket_name)
        try:
            self.client.fput_object(
                bucket_name=bucket_name,
                object_name=object_name,
                file_path=file_path,
            )
            logger.debug(f"Successfully uploaded {file_path} to {bucket_name}/{object_name}")
            return True
        except S3Error as e:
            logger.error(f"Failed to upload file to MinIO: {e}")
            return False

    def upload_bytes(self, bucket_name: str, object_name: str, data: bytes) -> bool:
        """Upload bytes data to MinIO.

        Args:
            bucket_name: Name of the bucket where the data will be stored.
            object_name: Object name (path of the file in MinIO).
            data: Bytes data to upload.

        Returns:
            True if upload is successful, False otherwise.
        """
        # Ensure bucket exists (deferred check)
        self._ensure_bucket_exists(bucket_name)
        try:
            from io import BytesIO
            data_stream = BytesIO(data)
            self.client.put_object(
                bucket_name=bucket_name,
                object_name=object_name,
                data=data_stream,
                length=len(data),
            )
            logger.debug(f"Successfully uploaded bytes to {bucket_name}/{object_name}")
            return True
        except S3Error as e:
            logger.error(f"Failed to upload bytes to MinIO: {e}")
            return False

    def download_file(self, bucket_name: str, object_name: str) -> bytes:
        """Download a file from MinIO and return its content as bytes.
        
        Note:
            This method loads the entire file content into memory at once.
            For large files, consider using streaming methods instead.
        
        Args:
            bucket_name: Name of the bucket containing the file.
            object_name: Object name (path of the file in MinIO).
            
        Returns:
            The file content as bytes. Returns empty bytes if download fails.
        """
        # Ensure bucket exists (deferred check)
        self._ensure_bucket_exists(bucket_name)
        response = None
        try:
            response = self.client.get_object(
                bucket_name=bucket_name,
                object_name=object_name,
            )
            data = response.read()  # Read entire file content into memory.
            logger.debug(f"Successfully retrieved {bucket_name}/{object_name}")
            return data
        except S3Error as e:
            logger.error(f"Failed to get file from MinIO: {e}")
            return b""
        finally:
            if response is not None:
                response.close()
                response.release_conn()

