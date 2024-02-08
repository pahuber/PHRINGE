from abc import ABC, abstractmethod


class BaseComponent(ABC):
    """Abstract base class representing the base component."""

    @abstractmethod
    def prepare(self, *args, **kwargs):
        """Prepare the component for the simulation.

        :param args: Additional arguments
        :param kwargs: Additional keyword arguments
        """
        pass
