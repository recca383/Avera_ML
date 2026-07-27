from app.core.config import Settings


def test_settings_allows_missing_azure_storage_config(monkeypatch):
    monkeypatch.delenv("AZURE_STORAGE_ACCOUNT_URL", raising=False)
    monkeypatch.delenv("AZURE_STORAGE_CONTAINER", raising=False)

    settings = Settings()

    assert settings.AZURE_STORAGE_ACCOUNT_URL == ""
    assert settings.AZURE_STORAGE_CONTAINER == ""
