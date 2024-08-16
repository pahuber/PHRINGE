import importlib
from pathlib import Path

from phringe.io.yaml_handler import YAMLHandler


def get_dict_from_path(file_path: Path) -> dict:
    """Read the dictionary from the path and return it.

    :param file_path: The path to the file
    :return: The dictionary
    """
    dict = YAMLHandler().read(file_path)
    return dict


def load_config(path):
    spec = importlib.util.spec_from_file_location("config", path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config
