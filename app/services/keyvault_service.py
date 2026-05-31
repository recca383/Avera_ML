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
    settings = get_settings()

    if not settings.AZURE_KEYVAULT_URL:
        logger.info("AZURE_KEYVAULT_URL not set — skipping Key Vault secret loading.")
        return

    vault_url = settings.AZURE_KEYVAULT_URL.rstrip("/")  # normalise

    logger.info("Loading secrets from Key Vault", keyvault_url=vault_url)

    credential = DefaultAzureCredential(
         exclude_managed_identity_credential=True
    )
    client = SecretClient(vault_url=vault_url, credential=credential)

    secret_map: dict[str, str] = {
        "INTERNAL_API_KEY": settings.KV_SECRET_INTERNAL_API_KEY,
    }

    try:
        for setting_field, kv_secret_name in secret_map.items():
            if not kv_secret_name:
                logger.warning("Secret name is empty", field=setting_field)
                continue

            # Log the exact name being requested — catches case/underscore mismatches
            logger.info(
                "Requesting secret",
                field=setting_field,
                kv_secret_name=kv_secret_name,
                vault_url=vault_url,
            )

            try:
                secret = await client.get_secret(kv_secret_name)
                object.__setattr__(settings, setting_field, secret.value or "")
                logger.info("Loaded secret", field=setting_field, kv_secret=kv_secret_name)
            except ResourceNotFoundError as exc:
                logger.warning(
                    "Secret not found — verify the name exists in the vault "
                    "and the Managed Identity has 'Key Vault Secrets User' at vault scope",
                    kv_secret_name=kv_secret_name,
                    vault_url=vault_url,
                    error=str(exc),   # SDK message often includes the attempted URL
                )
            except AzureError as exc:
                logger.error(
                    "Failed to retrieve secret",
                    kv_secret_name=kv_secret_name,
                    error=str(exc),
                )
    finally:
        await client.close()
        await credential.close()