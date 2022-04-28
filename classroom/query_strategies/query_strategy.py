from abc import ABC, abstractmethod
from typing import Literal, Type
from ..pref_graph import PrefGraph


_strategy_registry: dict[str, Type['QueryStrategy']] = {}

class QueryStrategy(ABC):
    """A strategy for selecting pairs of clips to query the user about."""
    def __init_subclass__(cls):
        try:
            name = getattr(cls, 'short_name')
        except AttributeError as e:
            raise TypeError(f"{cls.__name__} must define a short_name class field.") from e
        
        _strategy_registry[name] = cls
    
    @classmethod
    @property
    def available_strategies(cls) -> list[str]:
        """Return a list of all registered query strategies."""
        return list(_strategy_registry.keys())
    
    @classmethod
    def from_name(cls, name: str, graph: PrefGraph) -> 'QueryStrategy':
        return _strategy_registry[name](graph)

    @abstractmethod
    def __init__(self, graph: PrefGraph):
        ...

    @property
    @abstractmethod
    def current_query(self) -> tuple[str, str]:
        """
        Return the current pair of clips that the user should be queried about.
        The value of this property should only change after a call to `register_feedback`.
        """

    @property
    @abstractmethod
    def done(self) -> bool:
        """Whether the strategy has run out of pairs to query."""
    
    @abstractmethod
    def register_feedback(self, feedback: Literal['>', '<', '=']) -> None:
        """Callback for registering user's feedback regarding the query clips."""
