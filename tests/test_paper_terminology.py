from core.config import ConfigManager
from core.model_registry import get_model


def test_ncf_is_the_registered_paper_facing_model_name():
    model_class = get_model("NCF")
    config = ConfigManager("NCF", "ShortVideoSampled")

    assert model_class.__name__ == "NCF"
    assert config["model"] == "NCF"
    assert config["embedding_size"] == 512
    assert config["lcds"]["enabled"] is True
