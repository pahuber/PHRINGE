import importlib


def load_config(path):
    spec = importlib.util.spec_from_file_location("config", path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config
