from abc import ABC, abstractmethod


class BaseOSSClient(ABC):
    """Base OSS client class that defines a unified interface for object storage operations."""
    
    @abstractmethod
    def upload_file(self, bucket_name: str, object_name: str, file_path: str) -> bool:
        """Upload a file to OSS.
        
        Args:
            bucket_name: Name of the bucket where the file will be stored.
            object_name: Object name (path of the file in OSS).
            file_path: Local file path to upload.
            
        Returns:
            True if upload is successful, False otherwise.
        """
        pass

    
    @abstractmethod
    def download_file(self, bucket_name: str, object_name: str) -> bytes:
        """Download a file from OSS.
        
        Args:
            bucket_name: Name of the bucket containing the file.
            object_name: Object name (path of the file in OSS).
            
        Returns:
            The file content as bytes.
        """
        pass

    @abstractmethod
    def upload_bytes(self, bucket_name, object_name, data):
        """Upload bytes data to OSS.

        Args:
            bucket_name: Name of the bucket where the data will be stored.
            object_name: Object name (path of the file in OSS).
            data: Bytes data to upload.

        Returns:
            True if upload is successful, False otherwise.
        """
        pass

