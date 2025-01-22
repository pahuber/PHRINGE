import importlib
from pathlib import Path

from phringe.io.utils import load_config


class Configuration():
    def __init__(self, path):
        super().__init__()
        self.dict = load_config(path)

    def load_config(path: Path):
        spec = importlib.util.spec_from_file_location("config", path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        return config_module.config
