from pathlib import Path

import yaml


class YAMLReader:
    """Class representation of the YAML reader.
    """

    def read(self, file_path: Path) -> dict:
        """Read a YAML file and return its content as a dictionary.

        :param file_path: The path to the YAML file
        :return: The content of the YAML file as a dictionary
        """
        with open(file_path, 'r') as file:
            dict = yaml.load(file, Loader=yaml.SafeLoader)
        return dict
