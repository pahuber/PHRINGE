import importlib
from pathlib import Path

from phringe.core.base_entity import BaseEntity


class Configuration(BaseEntity):
    """Class representing the configuration of the simulation.

    Parameters
    ----------
    path : Path or None
        The path to the configuration file. If None, the config_dict parameter is used.
    config_dict : dict or None
        The configuration dictionary. If None, the path parameter is used.
    """
    path: Path = None
    config_dict: dict = None

    def __init__(self, path: Path = None, config_dict: dict = None):
        super().__init__()
        self.config_dict = self._load_config(path) if path is not None else config_dict

    @staticmethod
    def _load_config(path: Path):
        spec = importlib.util.spec_from_file_location("config", path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        return config_module.config
