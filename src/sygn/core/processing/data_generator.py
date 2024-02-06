from src.sygn.core.entities.observation import Observation
from src.sygn.core.entities.observatory.observatory import Observatory
from src.sygn.core.entities.scene import Scene
from src.sygn.core.entities.settings import Settings


class DataGenerator():
    def __init__(self,
                 settings: Settings,
                 observation: Observation,
                 observatory: Observatory,
                 system: Scene):
        self.settings = settings
        self.observation = observation
        self.observatory = observatory
        self.system = system

    def run(self) -> dict:
        # ...
        pass
