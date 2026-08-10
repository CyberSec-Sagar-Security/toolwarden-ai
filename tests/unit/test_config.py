import importlib
import os


def test_model_root_defaults_to_apps_and_models_path(monkeypatch):
    monkeypatch.delenv("TOOLWARDEN_MODEL_DIR", raising=False)
    from toolwarden import config

    importlib.reload(config)

    assert "Apps and Models" in str(config.MODEL_ROOT)
    assert "ToolWarden" in str(config.MODEL_ROOT)


def test_model_root_respects_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("TOOLWARDEN_MODEL_DIR", str(tmp_path))
    from toolwarden import config

    importlib.reload(config)

    assert config.MODEL_ROOT == tmp_path


def test_configure_hf_cache_env_sets_hf_home(monkeypatch, tmp_path):
    monkeypatch.setenv("TOOLWARDEN_MODEL_DIR", str(tmp_path))
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)
    from toolwarden import config

    importlib.reload(config)
    config.configure_hf_cache_env()

    assert os.environ["HF_HOME"] == str(config.HF_CACHE_DIR)
    assert config.HF_CACHE_DIR.exists()
