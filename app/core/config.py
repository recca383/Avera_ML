"""
Core configuration module.
Centralises all environment-driven settings using Pydantic BaseSettings.

Authentication model — Azure Active Directory / Managed Identity
----------------------------------------------------------------
This service authenticates to Azure Storage and Key Vault using
DefaultAzureCredential, which resolves credentials in this order:
  1. System-assigned or user-assigned Managed Identity  (production — Azure Container Apps)
  2. Azure CLI credentials  (`az login`)                (local development)
  3. Environment credentials (CI/CD pipelines)

NO storage connection strings or account keys are used anywhere.
Secrets that cannot be expressed as plain environment variables
(e.g. INTERNAL_API_KEY) can optionally be pulled from Key Vault
at startup via the same Managed Identity.
"""

from functools import lru_cache
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Application ────────────────────────────────────────────────────────────
    APP_NAME: str = "Signature Verification Inference Service"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "production"   # development | staging | production
    DEBUG: bool = False

    # ── Server ─────────────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1   # Keep at 1; the ML model singleton is not fork-safe

    # ── Azure Blob Storage — Managed Identity auth ──────────────────────────────
    # Use the blob service endpoint URL, NOT a connection string.
    # Format: https://<storage-account-name>.blob.core.windows.net
    # The Container App Managed Identity must be granted the role:
    #   "Storage Blob Data Contributor"  on the storage account (or container).
    # Leave these empty for local-only development; the service will still start
    # and report a clear error when a blob operation is attempted without Azure config.
    AZURE_STORAGE_ACCOUNT_URL: str = ""
    AZURE_STORAGE_CONTAINER: str = ""

    # ── Azure Key Vault — Managed Identity auth ─────────────────────────────────
    # When set, secrets are fetched from Key Vault at startup via Managed Identity.
    # The Container App Managed Identity must be granted the role:
    #   "Key Vault Secrets User"  on the Key Vault.
    # Leave empty to skip Key Vault integration (all config via env vars only).
    AZURE_KEYVAULT_URL: str = ""      # e.g. https://sig-keyvault.vault.azure.net

    # ── Optional: internal shared-secret (read from Key Vault if KV URL is set) ─
    # Name of the Key Vault secret that holds the shared-secret value.
    # The resolved value is written into INTERNAL_API_KEY at startup.
    KV_SECRET_INTERNAL_API_KEY: str = "sig-inference-internal-api-key"
    INTERNAL_API_KEY: str = ""        # populated at startup from Key Vault or env var

    # ── Model ──────────────────────────────────────────────────────────────────
    MODEL_PATH: str = "app/ml/exported_model/siamese_signature_model.pt"
    MODEL_INPUT_SIZE: int = 224         # expected H=W after preprocessing
    INFERENCE_THRESHOLD: float = 0.485123

    # ── Grad-CAM ───────────────────────────────────────────────────────────────
    GRADCAM_OUTPUT_PREFIX: str = "gradcam-output"
    GRADCAM_ALPHA: float = 0.4          # heatmap blend strength (0.0–1.0)

    # ── CORS ───────────────────────────────────────────────────────────────────
    CORS_ORIGINS: List[str] = ["*"]     # restrict to .NET API hostname in production
    CORS_METHODS: List[str] = ["POST", "GET", "OPTIONS"]
    CORS_HEADERS: List[str] = ["*"]

    # ── Logging ────────────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"            # json (production) | text (development)

    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = {"development", "staging", "production"}
        if v not in allowed:
            raise ValueError(f"ENVIRONMENT must be one of {allowed}")
        return v

    @field_validator("AZURE_STORAGE_ACCOUNT_URL", "AZURE_STORAGE_CONTAINER")
    @classmethod
    def validate_azure_storage_settings(cls, v: str, info) -> str:
        if not v:
            return ""
        if info.field_name == "AZURE_STORAGE_ACCOUNT_URL" and not v.startswith("http"):
            raise ValueError("AZURE_STORAGE_ACCOUNT_URL must be a valid URL when provided")
        return v

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached Settings singleton.
    lru_cache ensures the .env file is parsed exactly once for the process lifetime.
    """
    return Settings()
