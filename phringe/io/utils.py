from pathlib import Path

from phringe.io.txt_reader import TXTReader
from phringe.io.yaml_handler import YAMLHandler
from phringe.util.helpers import SpectrumContext


def get_dict_from_path(file_path: Path) -> dict:
    """Read the dictionary from the path and return it.

    :param file_path: The path to the file
    :return: The dictionary
    """
    dict = YAMLHandler().read(file_path)
    return dict


def get_spectra_from_path(spectrum_tuple: tuple[tuple[str, Path]]) -> list[SpectrumContext]:
    """Read the spectra from the paths and return a list of SpectrumContext objects.

    :param spectrum_tuple: List of tuples containing the planet name and the path to the corresponding spectrum text file
    :return: The spectra
    """
    try:
        spectrum_list = []
        for index_path, (planet_name, spectrum_file_path) in enumerate(spectrum_tuple):
            spectrum_list.append(SpectrumContext(planet_name, *TXTReader().read(Path(spectrum_file_path))))
    except TypeError:
        pass
    return spectrum_list
