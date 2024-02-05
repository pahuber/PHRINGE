from pathlib import Path
import yaml


class YAMLReader:
    """Read YAML files."""

    def read(self, file_path: Path) -> dict:
        """Read a YAML file and return its content as a dictionary."""
        with open(file_path, 'r') as file:
            dict = yaml.load(file, Loader=yaml.SafeLoader)
        return dict