from pathlib import Path

import yaml


class YAMLHandler:
    """Class representation of the YAML handler.
    """

    def read(self, file_path: Path) -> dict:
        """Read a YAML file and return its content as a dictionary.

        :param file_path: The path to the YAML file
        :return: The content of the YAML file as a dictionary
        """
        with open(file_path, 'r') as file:
            dict = yaml.load(file, Loader=yaml.SafeLoader)
        return dict

    def write(self, dict: dict, file_path: Path):
        """Write a dictionary to a YAML file.

        :param dict: The dictionary to be written to a YAML file
        :param file_path: The path to the YAML file
        """
        with open(file_path, 'w') as file:
            yaml.dump(dict, file)
