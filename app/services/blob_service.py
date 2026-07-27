"""
Azure Blob Storage service.
Authenticates exclusively via Azure Active Directory (Managed Identity).
No connection strings or storage account keys are used.

How authentication works
------------------------
DefaultAzureCredential tries credential sources in order:
  1. Managed Identity (system-assigned on the Container App) — production path
  2. Azure CLI  (`az login`)                                 — local dev path
  3. Environment variables (AZURE_CLIENT_ID / SECRET / TENANT_ID)

The Container App's Managed Identity must be assigned the RBAC role:
  "Storage Blob Data Contributor"
on the storage account or the specific container.

All images are downloaded and uploaded entirely in-memory — no temp files.
"""

import io
from typing import Optional

from azure.core.exceptions import ResourceNotFoundError, AzureError
from azure.identity.aio import DefaultAzureCredential
from azure.storage.blob import ContentSettings
from azure.storage.blob.aio import BlobServiceClient

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class BlobStorageService:
    """
    Async wrapper around the Azure Blob Storage SDK using AD token auth.
    Initialised once at app startup and shared for the process lifetime.
    """

    def __init__(self) -> None:
        self._client: Optional[BlobServiceClient] = None
        self._credential: Optional[DefaultAzureCredential] = None
        self._container: Optional[str] = None

    async def initialise(self) -> None:
        """
        Create the async BlobServiceClient authenticated via DefaultAzureCredential.
        Called during FastAPI lifespan startup — before any request is served.
        """
        settings = get_settings()
        self._container = settings.AZURE_STORAGE_CONTAINER

        if not settings.AZURE_STORAGE_ACCOUNT_URL:
            logger.warning(
                "Azure Blob Storage account URL is not configured; blob operations will fail until it is set",
                container=self._container,
            )
            return

        # DefaultAzureCredential is constructed once and reused.
        # It holds a token cache and handles transparent token refresh.
        self._credential = DefaultAzureCredential()

        self._client = BlobServiceClient(
            account_url=settings.AZURE_STORAGE_ACCOUNT_URL,
            credential=self._credential,
        )

        logger.info(
            "Azure Blob Storage client initialised (Managed Identity auth)",
            account_url=settings.AZURE_STORAGE_ACCOUNT_URL,
            container=self._container,
        )

    async def close(self) -> None:
        """Gracefully close the HTTP session and credential token cache."""
        if self._client:
            await self._client.close()
        if self._credential:
            await self._credential.close()
        logger.info("Azure Blob Storage client closed")

    async def download_blob(self, blob_id: str) -> bytes:
        """
        Download a blob by name and return its content as raw bytes.
        Raises FileNotFoundError when the blob does not exist.
        """
        self._assert_ready()

        try:
            blob_client = self._client.get_blob_client(
                container=self._container, blob=blob_id
            )
            download_stream = await blob_client.download_blob()
            data: bytes = await download_stream.readall()
            logger.debug("Downloaded blob", blob_id=blob_id, size_bytes=len(data))
            return data

        except ResourceNotFoundError:
            logger.warning("Blob not found", blob_id=blob_id, container=self._container)
            raise FileNotFoundError(
                f"Blob '{blob_id}' not found in container '{self._container}'."
            )
        except AzureError as exc:
            logger.error("Azure error downloading blob", blob_id=blob_id, error=str(exc))
            raise RuntimeError(f"Failed to download blob '{blob_id}': {exc}") from exc

    async def upload_blob(
        self,
        blob_id: str,
        data: bytes,
        content_type: str = "image/png",
        overwrite: bool = True,
    ) -> str:
        """
        Upload raw bytes to Azure Blob Storage.
        Returns the blob_id of the newly created blob.
        """
        self._assert_ready()

        try:
            blob_client = self._client.get_blob_client(
                container=self._container, blob=blob_id
            )
            await blob_client.upload_blob(
                io.BytesIO(data),
                blob_type="BlockBlob",
                overwrite=overwrite,
                content_settings=ContentSettings(content_type=content_type),
            )
            logger.info(
                "Uploaded blob",
                blob_id=blob_id,
                container=self._container,
                size_bytes=len(data),
            )
            return blob_id

        except AzureError as exc:
            logger.error("Azure error uploading blob", blob_id=blob_id, error=str(exc))
            raise RuntimeError(f"Failed to upload blob '{blob_id}': {exc}") from exc

    def _assert_ready(self) -> None:
        settings = get_settings()
        if not settings.AZURE_STORAGE_ACCOUNT_URL:
            raise RuntimeError(
                "Azure Blob Storage is not configured. Set AZURE_STORAGE_ACCOUNT_URL and "
                "AZURE_STORAGE_CONTAINER before attempting blob operations."
            )
        if self._client is None:
            raise RuntimeError(
                "BlobStorageService has not been initialised. "
                "Ensure initialise() is called during application startup."
            )


# ── Module-level singleton ─────────────────────────────────────────────────────
blob_storage_service = BlobStorageService()


def get_blob_storage_service() -> BlobStorageService:
    """FastAPI dependency: returns the application-scoped BlobStorageService."""
    return blob_storage_service
