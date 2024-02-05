from sygn.src.sygn.core.entities.observation import Observation
from sygn.src.sygn.core.entities.observatory.observatory import Observatory
from sygn.src.sygn.core.entities.settings import Settings
from sygn.src.sygn.core.entities.system import System


class DataGenerator():
    def __init__(self,
                 settings: Settings,
                 observation: Observation,
                 observatory: Observatory,
                 system: System):
        self.settings = settings
        self.observation = observation
        self.observatory = observatory
        self.system = system

    def run(self) -> dict:
        # ...
        pass