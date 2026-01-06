import pytest

from api import config


def test_config_workflow(tmp_path):
    # Mock CONFIG_FILE to use tmp_path
    mock_config_dir = tmp_path / ".config" / "hn_rerank"
    mock_config_file = mock_config_dir / "config.json"

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "CONFIG_DIR", mock_config_dir)
        mp.setattr(config, "CONFIG_FILE", mock_config_file)

        # 1. Load non-existent config
        assert config.load_config() == {}
        assert config.get_username() is None

        # 2. Save config
        config.save_config("username", "testuser")
        assert mock_config_file.exists()

        # 3. Load config
        loaded = config.load_config()
        assert loaded["username"] == "testuser"
        assert config.get_username() == "testuser"

        # 4. Save another key
        config.save_config("model", "bge")
        assert config.load_config()["model"] == "bge"
        assert config.load_config()["username"] == "testuser"

def test_load_corrupt_config(tmp_path):
    mock_config_dir = tmp_path / "corrupt"
    mock_config_file = mock_config_dir / "config.json"
    mock_config_dir.mkdir()
    mock_config_file.write_text("invalid json{")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "CONFIG_DIR", mock_config_dir)
        mp.setattr(config, "CONFIG_FILE", mock_config_file)

        assert config.load_config() == {}
