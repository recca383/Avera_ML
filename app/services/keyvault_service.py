"""
Azure Key Vault service.
Fetches secrets at application startup using Managed Identity (DefaultAzureCredential).

Why pull from Key Vault at startup rather than at request time?
---------------------------------------------------------------
- Secrets are loaded once and kept in the Settings object — no per-request latency.
- The Container App's Managed Identity needs only "Key Vault Secrets User" role.
- No secret values are ever written to environment variables visible outside
  this process or to any log output.

Required Azure RBAC assignment
-------------------------------
  Principal : Container App system-assigned Managed Identity
  Role      : Key Vault Secrets User
  Scope     : /subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.KeyVault/vaults/<kv-name>

If AZURE_KEYVAULT_URL is not set in the environment, this module is a no-op.
"""

from azure.core.exceptions import ResourceNotFoundError, AzureError
from azure.identity.aio import DefaultAzureCredential
from azure.keyvault.secrets.aio import SecretClient

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


async def load_secrets_from_keyvault() -> None:
    """
    Fetch secrets from Key Vault and write them into the cached Settings object.
    Must be called during FastAPI lifespan startup, before any request is served.

    Currently managed secrets
    -------------------------
    - INTERNAL_API_KEY  ← Key Vault secret named by settings.KV_SECRET_INTERNAL_API_KEY
      Used by security.py to validate the X-Internal-Api-Key header from the .NET API.

    Add additional secrets to the mapping dict below as the service grows.
    """
    settings = get_settings()

    if not settings.AZURE_KEYVAULT_URL:
        logger.info(
            "AZURE_KEYVAULT_URL not set — skipping Key Vault secret loading. "
            "Ensure all required secrets are supplied via environment variables."
        )
        return

    logger.info("Loading secrets from Key Vault", keyvault_url=settings.AZURE_KEYVAULT_URL)

    credential = DefaultAzureCredential()
    client = SecretClient(vault_url=settings.AZURE_KEYVAULT_URL, credential=credential)

    logger.info("Log in to Azure Key Vault successful, fetching secrets")
    # Map: Settings field name → Key Vault secret name
    secret_map: dict[str, str] = {
        "INTERNAL_API_KEY": settings.KV_SECRET_INTERNAL_API_KEY,
    }

    try:
        for setting_field, kv_secret_name in secret_map.items():
            if not kv_secret_name:
                continue
            try:
                secret = await client.get_secret(kv_secret_name)
                # Directly mutate the cached Settings instance.
                # pydantic-settings models are not frozen, so this is safe.
                object.__setattr__(settings, setting_field, secret.value or "")
                logger.info(
                    "Loaded secret from Key Vault",
                    field=setting_field,
                    kv_secret=kv_secret_name,
                )
            except ResourceNotFoundError:
                logger.warning(
                    "Key Vault secret not found — field will use its default/env value",
                    kv_secret=kv_secret_name,
                    field=setting_field,
                )
            except AzureError as exc:
                # Log and continue — a missing optional secret should not crash startup.
                logger.error(
                    "Failed to retrieve Key Vault secret",
                    kv_secret=kv_secret_name,
                    field=setting_field,
                    error=str(exc),
                )
    finally:
        await client.close()
        await credential.close()

    logger.info("Key Vault secret loading complete")
